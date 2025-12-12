// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Based on singleflight-async by ihciah
// Original: https://github.com/ihciah/singleflight-async
// Licensed under MIT/Apache-2.0

//! Deduplicates async tasks into a single execution.
//!
//! This crate provides [`UniFlight`], a mechanism for deduplicating concurrent async operations.
//! When multiple tasks request the same work (identified by a key), only the first task (the
//! "leader") performs the actual work while subsequent tasks (the "followers") wait and receive
//! a clone of the result.
//!
//! # When to Use
//!
//! Use `UniFlight` when you have expensive or rate-limited operations that may be requested
//! concurrently with the same parameters:
//!
//! - **Cache population**: Prevent thundering herd when a cache entry expires
//! - **API calls**: Deduplicate concurrent requests to the same endpoint
//! - **Database queries**: Coalesce identical queries issued simultaneously
//! - **File I/O**: Avoid reading the same file multiple times concurrently
//!
//! # Example
//!
//! ```
//! use uniflight::UniFlight;
//!
//! # async fn example() {
//! let group: UniFlight<&str, String> = UniFlight::new();
//!
//! // Multiple concurrent calls with the same key will share a single execution
//! let result = group.work(&"user:123", || async {
//!     // This expensive operation runs only once, even if called concurrently
//!     "expensive_result".to_string()
//! }).await;
//! # }
//! ```
//!
//! # Cancellation and Panic Safety
//!
//! `UniFlight` handles task cancellation and panics gracefully:
//!
//! - If the leader task is cancelled or dropped, a follower becomes the new leader
//! - If the leader task panics, a follower becomes the new leader and executes its work
//! - Followers that join before the leader completes receive the cached result
//!
//! # Thread Safety
//!
//! [`UniFlight`] is `Send` and `Sync`, and can be shared across threads. The returned futures
//! do not require `Send` bounds on the closure or its output.

#![doc(html_logo_url = "https://media.githubusercontent.com/media/microsoft/oxidizer/refs/heads/main/crates/uniflight/logo.png")]
#![doc(html_favicon_url = "https://media.githubusercontent.com/media/microsoft/oxidizer/refs/heads/main/crates/uniflight/favicon.ico")]

use std::{
    collections::HashMap,
    hash::Hash,
    mem::replace,
    pin::Pin,
    sync::{Arc as SyncArc, Weak},
    task::{Context, Poll},
};

use parking_lot::Mutex as SyncMutex;
use thread_aware::{
    Arc, PerThread, ThreadAware,
    affinity::{MemoryAffinity, PinnedAffinity},
    storage::Strategy,
};
use tokio::sync::Mutex as AsyncMutex;

type SharedMapping<K, T, S> = Arc<SyncMutex<HashMap<K, BroadcastOnce<T>>>, S>;

/// Represents a class of work and creates a space in which units of work
/// can be executed with duplicate suppression.
#[derive(Debug, ThreadAware)]
pub struct UniFlight<K, T, S = PerThread>
where
    S: Strategy,
{
    mapping: SharedMapping<K, T, S>,
}

impl<K, T, S> Default for UniFlight<K, T, S>
where
    K: Send + 'static,
    T: Send + 'static,
    S: Strategy,
{
    fn default() -> Self {
        Self {
            mapping: Arc::new(SyncMutex::default),
        }
    }
}

struct Shared<T> {
    slot: AsyncMutex<Option<T>>,
}

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self {
            slot: AsyncMutex::new(None),
        }
    }
}

/// `BroadcastOnce` stores a weak reference to the shared slot in the mapping.
/// Leaders hold the strong reference; followers upgrade from this weak reference.
#[derive(Clone)]
struct BroadcastOnce<T> {
    shared: Weak<Shared<T>>,
}

impl<T> BroadcastOnce<T> {
    fn new() -> (Self, SyncArc<Shared<T>>) {
        let shared = SyncArc::new(Shared::default());
        (
            Self {
                shared: SyncArc::downgrade(&shared),
            },
            shared,
        )
    }
}

/// State machine for the waiter future.
enum WaiterState<K, T, F, S>
where
    S: Strategy,
{
    /// Initial state - hasn't been polled yet.
    /// The `shared` ref is always present (created at `work()` time).
    /// `is_leader` determines whether we run `func` or wait for another's result.
    Pending {
        func: F,
        key: K,
        mapping: SharedMapping<K, T, S>,
        shared: SyncArc<Shared<T>>,
        is_leader: bool,
    },
    /// Leader: running `do_work`
    Leading { future: Pin<Box<dyn Future<Output = T> + Send>> },
    /// Follower: waiting for leader's result, no cleanup
    Following { future: Pin<Box<dyn Future<Output = T> + Send>> },
    /// Relocated while running - just finish and return result
    Detached { future: Pin<Box<dyn Future<Output = T> + Send>> },
    /// Terminal state
    Completed,
}

/// Leader's async work: acquire lock, compute value, store it.
async fn do_work<T, F, Fut>(shared: SyncArc<Shared<T>>, func: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
    T: Clone,
{
    let mut slot = shared.slot.lock().await;
    if let Some(value) = slot.as_ref() {
        return value.clone();
    }

    let value = func().await;
    *slot = Some(value.clone());

    value
}

/// Future returned by [`UniFlight::work`] that resolves to the work result.
///
/// This future implements a state machine for coordinating work deduplication
/// and supports relocation between thread affinities via [`ThreadAware`].
pub struct BroadcastOnceWaiter<K, T, F, S>
where
    S: Strategy,
{
    state: WaiterState<K, T, F, S>,
}

impl<K, T, F, S> std::fmt::Debug for BroadcastOnceWaiter<K, T, F, S>
where
    S: Strategy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BroadcastOnceWaiter")
    }
}

// BroadcastOnceWaiter can be Unpin because:
// - The struct itself has no self-referential data
// - The only pinned data is inside Pin<Box<...>> which is itself Unpin
//   (the Box provides a stable heap location for the future)
// - Moving BroadcastOnceWaiter doesn't move the boxed future
impl<K, T, F, S: Strategy> Unpin for BroadcastOnceWaiter<K, T, F, S> {}

impl<T> std::fmt::Debug for BroadcastOnce<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BroadcastOnce")
    }
}

impl<K, T, F, Fut, S> Future for BroadcastOnceWaiter<K, T, F, S>
where
    K: Hash + Eq + Clone + Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
    T: Clone + Send + 'static,
    S: Strategy,
{
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        // Safe because BroadcastOnceWaiter implements Unpin
        let this = self.as_mut().get_mut();

        // Pending: transition to Leading or Following based on is_leader
        if matches!(this.state, WaiterState::Pending { .. }) {
            let WaiterState::Pending {
                func, shared, is_leader, ..
            } = replace(&mut this.state, WaiterState::Completed)
            else {
                unreachable!("state changed unexpectedly");
            };

            // Try fast path first: maybe value is already ready
            if let Ok(guard) = shared.slot.try_lock()
                && let Some(value) = guard.as_ref()
            {
                return Poll::Ready(value.clone());
            }

            // Value not ready - run do_work which handles both cases:
            // - Normal leader: run func, store value
            // - Follower (leader still working): wait for mutex, then return cached value
            // - Promoted follower (leader failed): wait for mutex, slot empty, run func
            let future: Pin<Box<dyn Future<Output = T> + Send>> = Box::pin(do_work(shared, func));
            this.state = if is_leader {
                WaiterState::Leading { future }
            } else {
                WaiterState::Following { future }
            };
        }

        // Poll the current state
        match &mut this.state {
            WaiterState::Leading { future, .. } => match future.as_mut().poll(cx) {
                Poll::Ready(value) => {
                    // Don't remove from mapping - late followers may still need to upgrade.
                    // The Weak reference naturally becomes stale when all strong refs are dropped.
                    // New callers after that will fail to upgrade and become new leaders.
                    this.state = WaiterState::Completed;
                    Poll::Ready(value)
                }
                Poll::Pending => Poll::Pending,
            },

            WaiterState::Following { future } | WaiterState::Detached { future } => match future.as_mut().poll(cx) {
                Poll::Ready(value) => {
                    this.state = WaiterState::Completed;
                    Poll::Ready(value)
                }
                Poll::Pending => Poll::Pending,
            },

            WaiterState::Completed => unreachable!("polled after completion"),
            WaiterState::Pending { .. } => unreachable!("handled above"),
        }
    }
}

impl<K, T, F, Fut, S> ThreadAware for BroadcastOnceWaiter<K, T, F, S>
where
    K: Clone + Hash + Eq + Send + 'static,
    T: Clone + Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
    S: Strategy,
{
    fn relocated(self, source: MemoryAffinity, destination: PinnedAffinity) -> Self {
        match self.state {
            WaiterState::Pending {
                func,
                key,
                mapping,
                shared,
                ..
            } => {
                // Drop old shared ref (triggers leader election on old affinity if we were leader)
                drop(shared);

                // Relocate mapping to access new affinity's storage
                let mapping = mapping.relocated(source, destination);

                // Re-register on new affinity
                let mut map = mapping.lock();
                let (shared, is_leader) = if let Some(existing) = map.get(&key) {
                    if let Some(shared) = existing.shared.upgrade() {
                        // Leader exists on new affinity - become follower
                        (shared, false)
                    } else {
                        // Leader gone - become new leader
                        let (broadcast, shared) = BroadcastOnce::new();
                        map.insert(key.clone(), broadcast);
                        (shared, true)
                    }
                } else {
                    // No entry on new affinity - become leader
                    let (broadcast, shared) = BroadcastOnce::new();
                    map.insert(key.clone(), broadcast);
                    (shared, true)
                };
                drop(map);

                Self {
                    state: WaiterState::Pending {
                        func,
                        key,
                        mapping,
                        shared,
                        is_leader,
                    },
                }
            }

            WaiterState::Leading { future, .. } | WaiterState::Following { future } => {
                // Mid-execution: let the future complete without coordination
                Self {
                    state: WaiterState::Detached { future },
                }
            }

            WaiterState::Detached { .. } | WaiterState::Completed => self,
        }
    }
}

impl<K, T, S> UniFlight<K, T, S>
where
    K: Hash + Eq + Send + 'static,
    T: Send + 'static,
    S: Strategy,
{
    /// Creates a new `UniFlight` instance.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute and return the value for a given function, making sure that only one
    /// operation is in-flight at a given moment. If a duplicate call comes in, that caller will
    /// wait until the original call completes and return the same value.
    ///
    /// Leader/follower role is determined at call time:
    /// - If no entry exists, create one and become leader
    /// - If entry exists and can upgrade, become follower
    /// - If entry exists but can't upgrade (leader gone), replace and become leader
    pub fn work<F, Fut>(&self, key: &K, func: F) -> BroadcastOnceWaiter<K, T, F, S>
    where
        K: Clone,
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
        T: Clone,
    {
        let mut map = self.mapping.lock();
        let (shared, is_leader) = if let Some(existing) = map.get(key) {
            if let Some(shared) = existing.shared.upgrade() {
                // Leader exists - become follower
                (shared, false)
            } else {
                // Leader gone - become new leader
                let (broadcast, shared) = BroadcastOnce::new();
                map.insert(key.clone(), broadcast);
                (shared, true)
            }
        } else {
            // No entry - become leader
            let (broadcast, shared) = BroadcastOnce::new();
            map.insert(key.clone(), broadcast);
            (shared, true)
        };
        drop(map);

        BroadcastOnceWaiter {
            state: WaiterState::Pending {
                func,
                key: key.clone(),
                mapping: self.mapping.clone(),
                shared,
                is_leader,
            },
        }
    }
}
