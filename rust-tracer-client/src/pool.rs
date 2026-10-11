//! Connection pool to the control plane, ported from the ArmoniK load
//! balancer (ArmoniK.Infra.Plugins, `load-balancer/src/cluster.rs`: the
//! connection half of `Cluster`, without the load balancer's header
//! forwarding and availability tracking).
//!
//! Every request checks a connection out for its duration (a stream's whole
//! lifetime for streaming calls). Each connection carries a request budget
//! and is replaced by a new one once spent: kept below the HTTP/2 ingress's
//! per-connection request cap (nginx closes a connection after 1000 requests
//! by default), so the ingress never closes a connection under requests in
//! flight ("connection closed" failures).

use std::num::NonZeroUsize;
use std::ops::Deref;

use tokio::sync::{Semaphore, SemaphorePermit};

use crate::bag::Bag;

/// How the pool uses connections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolConfig {
    /// Maximum number of connections (default: 4 x tokio workers, 0 =
    /// unlimited).
    pub pool_size: Option<usize>,
    /// Requests sent on a connection before it is replaced (default: 800,
    /// 0 = never).
    pub requests_per_connection: Option<usize>,
    /// Whether a connection carries several requests at once.
    pub multiplex: bool,
}

pub struct Pool {
    config: armonik::ClientConfig,
    clients: Bag<(armonik::Client, Option<NonZeroUsize>)>,
    requests_per_connection: Option<NonZeroUsize>,
    semaphore: Option<Semaphore>,
    multiplex: bool,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("endpoint", &self.config.endpoint)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// Must be called within a tokio runtime (the default pool size follows
    /// its number of workers). Connects lazily: see [`Pool::client`].
    pub fn new(config: armonik::ClientConfig, pool: PoolConfig) -> Self {
        Self {
            config,
            clients: Bag::new(),
            // The default stays below common HTTP/2 ingress per-connection request caps
            // (nginx defaults to 1000) so upstreams never GOAWAY a connection mid-flight.
            requests_per_connection: NonZeroUsize::new(pool.requests_per_connection.unwrap_or(800)),
            // Empirical default: enough parallel connections to keep every runtime
            // worker busy without unbounded connection growth.
            semaphore: NonZeroUsize::new(
                pool.pool_size
                    .unwrap_or(4 * tokio::runtime::Handle::current().metrics().num_workers()),
            )
            .map(|size| Semaphore::new(size.get())),
            multiplex: pool.multiplex,
        }
    }

    /// A connection for one request (or one stream): held until the handle
    /// is dropped, unless multiplexed.
    pub async fn client(&self) -> Result<PooledClient<'_>, armonik::client::ConnectionError> {
        // Acquire a pool slot first (no semaphore = unbounded pool).
        let permit = match &self.semaphore {
            Some(semaphore) => Some(semaphore.acquire().await.unwrap()),
            None => None,
        };
        // Reuse an idle pooled connection if any, otherwise dial a new one with a fresh
        // request budget.
        let (client, requests) = match self.clients.pop() {
            Some(client) => client,
            None => {
                let client = armonik::Client::with_config(self.config.clone()).await?;
                (client, self.requests_per_connection)
            }
        };

        let mut client = PooledClient {
            client,
            requests,
            bag: Some(&self.clients),
            permit,
        };

        if self.multiplex {
            // Multiplexed mode: hand the connection back to the pool immediately (and
            // release the permit) so concurrent requests can share the same HTTP/2
            // connection while this one is still in flight.
            client.release();
        }

        Ok(client)
    }
}

/// A service client on a pooled connection, held as long as this value
/// lives: one per request (a stream's whole lifetime for streaming calls),
/// for the connection's request budget to count requests.
pub struct Pooled<'a, S> {
    service: S,
    // Dropped after the service client.
    _connection: PooledClient<'a>,
}

impl<S> Deref for Pooled<'_, S> {
    type Target = S;

    fn deref(&self) -> &S {
        &self.service
    }
}

impl<S> std::ops::DerefMut for Pooled<'_, S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.service
    }
}

macro_rules! services {
    ($($name:ident: $service:ident = $into:ident),* $(,)?) => {
        impl Pool {
            $(
                pub async fn $name(
                    &self,
                ) -> Result<Pooled<'_, armonik::client::$service<armonik::Client>>, armonik::client::ConnectionError> {
                    let connection = self.client().await?;
                    Ok(Pooled {
                        service: (*connection).clone().$into(),
                        _connection: connection,
                    })
                }
            )*
        }
    };
}

services!(
    results: Results = into_results,
    sessions: Sessions = into_sessions,
    tasks: Tasks = into_tasks,
    events: Events = into_events,
);

/// Pooled connection handle. On drop (or at acquisition in multiplexed mode), the
/// connection goes back into the bag with a decremented request budget; once the
/// budget hits zero it is discarded instead, forcing a reconnection.
pub struct PooledClient<'a> {
    client: armonik::Client,
    requests: Option<NonZeroUsize>,
    bag: Option<&'a Bag<(armonik::Client, Option<NonZeroUsize>)>>,
    permit: Option<SemaphorePermit<'a>>,
}

impl Deref for PooledClient<'_> {
    type Target = armonik::Client;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl Drop for PooledClient<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

impl PooledClient<'_> {
    /// Return the connection to the pool (dropping it when its request budget is
    /// exhausted) and release the pool slot. Idempotent.
    fn release(&mut self) {
        if let Some(bag) = self.bag.take() {
            match self.requests {
                Some(size) => {
                    if let Some(size) = NonZeroUsize::new(size.get() - 1) {
                        bag.push((self.client.clone(), Some(size)));
                    }
                }
                None => bag.push((self.client.clone(), None)),
            }
        }
        std::mem::drop(self.permit.take());
    }
}

#[cfg(test)]
mod tests {
    use armonik::reexports::tonic;

    use super::*;

    fn lazy_client() -> armonik::Client {
        // Lazy channel: nothing listens on the endpoint, and no request is sent.
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        armonik::Client::with_channel(channel)
    }

    fn budgets(pool: &Pool) -> Vec<Option<NonZeroUsize>> {
        std::iter::from_fn(|| pool.clients.pop())
            .map(|(_, requests)| requests)
            .collect()
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // crossbeam-epoch is currently incompatible with MIRI: https://github.com/crossbeam-rs/crossbeam/issues/1181
    async fn multiplexed_client_is_released_once() {
        let pool = Pool::new(
            armonik::ClientConfig::default(),
            PoolConfig {
                multiplex: true,
                ..Default::default()
            },
        );
        pool.clients.push((lazy_client(), NonZeroUsize::new(3)));

        // Released once at acquisition; dropping it must not release it again.
        drop(pool.client().await.unwrap());

        assert_eq!(budgets(&pool), [NonZeroUsize::new(2)]);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn spent_connections_are_dropped() {
        let pool = Pool::new(armonik::ClientConfig::default(), PoolConfig::default());
        pool.clients.push((lazy_client(), NonZeroUsize::new(1)));
        // The last request of its budget: not returned to the pool.
        drop(pool.client().await.unwrap());
        assert_eq!(budgets(&pool), []);

        // Without a budget, a connection is kept forever.
        pool.clients.push((lazy_client(), None));
        drop(pool.client().await.unwrap());
        assert_eq!(budgets(&pool), [None]);
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn the_pool_size_bounds_connections_in_use() {
        let pool = Pool::new(
            armonik::ClientConfig::default(),
            PoolConfig {
                pool_size: Some(1),
                ..Default::default()
            },
        );
        pool.clients.push((lazy_client(), NonZeroUsize::new(10)));
        let held = pool.client().await.unwrap();
        // The only slot is taken: the next checkout waits for it.
        let waiting = tokio::time::timeout(std::time::Duration::from_millis(50), pool.client());
        assert!(waiting.await.is_err());
        drop(held);
        assert!(pool.client().await.is_ok());
    }
}
