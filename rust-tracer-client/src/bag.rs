//! Copied from the ArmoniK load balancer
//! (ArmoniK.Infra.Plugins, `load-balancer/src/bag.rs`), where the connection
//! pool (`pool.rs`) comes from.

use std::sync::RwLock;

use crossbeam_deque::{Stealer, Worker};
use thread_local::ThreadLocal;

/// Unordered pool of `T` optimized for concurrent check-in/check-out.
///
/// Each thread lazily gets its own lock-free LIFO deque: pushes and pops touch only the
/// local deque on the hot path (no shared-lock contention), and LIFO order hands back the
/// most recently returned, still-warm item. Stealing from other threads' deques happens
/// only when the local one is empty. The bag is `Sync` with only `T: Send`.
///
/// Per-thread deques are registered on first use and never reclaimed, so the bag should
/// only be used from long-lived pool threads (tokio/rayon workers).
#[derive(Debug)]
pub struct Bag<T: Send> {
    worker: ThreadLocal<Worker<T>>,
    stealers: RwLock<Vec<Stealer<T>>>,
}

impl<T: Send> Default for Bag<T> {
    fn default() -> Self {
        Self {
            worker: Default::default(),
            stealers: Default::default(),
        }
    }
}

impl<T: Send> Bag<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get this thread's deque, registering its stealer on first use.
    fn worker(&self) -> &Worker<T> {
        self.worker.get_or(|| {
            let worker = Worker::new_lifo();
            self.stealers.write().unwrap().push(worker.stealer());
            worker
        })
    }

    pub fn pop(&self) -> Option<T> {
        let worker = self.worker();
        if let Some(item) = worker.pop() {
            Some(item)
        } else {
            // Local deque is empty: steal a batch from another thread. `Steal::Retry`
            // signals a race with the owner, so loop until every stealer reports
            // either Empty or Success.
            let mut retry = true;
            while retry {
                retry = false;
                for stealer in self.stealers.read().unwrap().iter() {
                    match stealer.steal_batch_and_pop(worker) {
                        crossbeam_deque::Steal::Empty => (),
                        crossbeam_deque::Steal::Success(item) => return Some(item),
                        crossbeam_deque::Steal::Retry => retry = true,
                    }
                }
            }
            None
        }
    }

    pub fn push(&self, item: T) {
        self.worker().push(item);
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread};

    use super::*;

    #[test]
    fn bag_auto_traits() {
        struct NotSync(#[allow(unused)] *mut i32);
        unsafe impl Send for NotSync {}

        let bag = Bag::<NotSync>::new();

        _ = &bag as &dyn Send;
        _ = &bag as &dyn Sync;
    }

    #[test]
    #[cfg_attr(miri, ignore)] // crossbeam-epoch is currently incompatible with MIRI: https://github.com/crossbeam-rs/crossbeam/issues/1181
    fn foo() {
        let bag = Bag::new();
        bag.push(1);
        bag.push(2);
        bag.push(3);

        let values = [
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
        ];

        thread::scope(|s| {
            s.spawn(|| {
                _ = values[1].lock().unwrap().insert(bag.pop());
            });
            s.spawn(|| {
                _ = values[2].lock().unwrap().insert(bag.pop());
            });
            s.spawn(|| {
                _ = values[3].lock().unwrap().insert(bag.pop());
            });
            thread::yield_now();
            _ = values[0].lock().unwrap().insert(bag.pop());
        });

        let mut values = values.map(|x| x.into_inner().unwrap().unwrap());
        values.sort();

        assert_eq!(values, [None, Some(1), Some(2), Some(3)]);
    }
}
