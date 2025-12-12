// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Integration tests for the uniflight crate.

use std::{
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::{AcqRel, Acquire},
        },
    },
    time::Duration,
};

use futures_util::{StreamExt, stream::FuturesUnordered};
use thread_aware::PerThread;
use uniflight::UniFlight;

fn unreachable_future() -> std::future::Pending<String> {
    std::future::pending()
}

#[tokio::test]
async fn direct_call() {
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let result = group
        .work(&"key", || async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            "Result".to_string()
        })
        .await;
    assert_eq!(result, "Result");
}

#[tokio::test]
async fn parallel_call() {
    let call_counter = Arc::new(AtomicUsize::default());

    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let futures = FuturesUnordered::new();
    for _ in 0..10 {
        let counter = Arc::clone(&call_counter);
        futures.push(group.work(&"key", move || async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            counter.fetch_add(1, AcqRel);
            "Result".to_string()
        }));
    }

    assert!(futures.all(|out| async move { out == "Result" }).await);
    assert_eq!(call_counter.load(Acquire), 1);
}

#[tokio::test]
async fn parallel_call_seq_await() {
    let call_counter = Arc::new(AtomicUsize::default());

    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let mut futures = Vec::new();
    for _ in 0..10 {
        let counter = Arc::clone(&call_counter);
        futures.push(group.work(&"key", move || async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            counter.fetch_add(1, AcqRel);
            "Result".to_string()
        }));
    }

    for fut in futures {
        assert_eq!(fut.await, "Result");
    }
    assert_eq!(call_counter.load(Acquire), 1);
}

#[tokio::test]
async fn call_with_static_str_key() {
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let key = "key".to_string();
    let result = group
        .work(&key, || async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            "Result".to_string()
        })
        .await;
    assert_eq!(result, "Result");
}

#[tokio::test]
async fn call_with_static_string_key() {
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let key = "key".to_string();
    let result = group
        .work(&key, || async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            "Result".to_string()
        })
        .await;
    assert_eq!(result, "Result");
}

#[tokio::test]
async fn call_with_custom_key() {
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct K(i32);
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let key = K(1);
    let result = group
        .work(&key, || async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            "Result".to_string()
        })
        .await;
    assert_eq!(result, "Result");
}

#[tokio::test]
async fn late_wait() {
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let key = "key".to_string();
    let fut_early = group.work(&key, || async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        "Result".to_string()
    });
    let fut_late = group.work(&key, unreachable_future);
    assert_eq!(fut_early.await, "Result");
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(fut_late.await, "Result");
}

#[tokio::test]
async fn cancel() {
    let group: UniFlight<_, _, PerThread> = UniFlight::new();
    let key = "key".to_string();

    // the executer cancelled and the other awaiter will create a new future and execute.
    let fut_cancel = group.work(&key, unreachable_future);
    let _ = tokio::time::timeout(Duration::from_millis(10), fut_cancel).await;
    let fut_late = group.work(&key, || async { "Result2".to_string() });
    assert_eq!(fut_late.await, "Result2");

    // the first executer is slow but not dropped, so the result will be the first ones.
    let begin = tokio::time::Instant::now();
    let fut_1 = group.work(&key, || async {
        tokio::time::sleep(Duration::from_millis(2000)).await;
        "Result1".to_string()
    });
    let fut_2 = group.work(&key, unreachable_future);
    let (v1, v2) = tokio::join!(fut_1, fut_2);
    assert_eq!(v1, "Result1");
    assert_eq!(v2, "Result1");
    assert!(begin.elapsed() > Duration::from_millis(1500));
}

#[tokio::test]
async fn leader_panic_in_spawned_task() {
    let call_counter = Arc::new(AtomicUsize::default());
    let group: Arc<UniFlight<String, String, PerThread>> = Arc::new(UniFlight::new());
    let key = Arc::new("key".to_string());

    // First task will panic in a spawned task (no catch_unwind)
    let group_clone = Arc::clone(&group);
    let key_clone = Arc::clone(&key);
    let handle = tokio::spawn(async move {
        group_clone
            .work(&key_clone, move || async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                panic!("leader panicked in spawned task");
                #[expect(unreachable_code, reason = "Required to satisfy return type after panic")]
                "never".to_string()
            })
            .await
    });

    // Give time for the spawned task to register and start
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Second task should become the new leader after the first panics
    let group_clone = Arc::clone(&group);
    let counter = Arc::clone(&call_counter);
    let fut_follower = group_clone.work(&key, move || async move {
        counter.fetch_add(1, AcqRel);
        "Result".to_string()
    });

    // Wait for the spawned task to panic
    let spawn_result = handle.await;
    assert!(spawn_result.is_err());

    // The follower should succeed - Rust's drop semantics ensure the mutex is released
    let result = fut_follower.await;
    assert_eq!(result, "Result");
    assert_eq!(call_counter.load(Acquire), 1);
}

#[tokio::test]
async fn debug_impl() {
    let group: UniFlight<String, String, PerThread> = UniFlight::new();
    let key = "key".to_string();

    // Test Debug on empty group
    let debug_str = format!("{:?}", group);
    assert!(debug_str.contains("UniFlight"));

    // Create a pending work item to populate the mapping with a BroadcastOnce
    let fut = group.work(&key, || async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        "Result".to_string()
    });

    // Debug should still work with entries in the mapping
    let debug_str = format!("{:?}", group);
    assert!(debug_str.contains("UniFlight"));
    assert!(debug_str.contains("BroadcastOnce"));

    // Complete the work
    assert_eq!(fut.await, "Result");
}
