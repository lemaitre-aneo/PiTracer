//! Waiting for results: an abstraction over the two ways ArmoniK reports
//! result completion, so the strategy is chosen at runtime.

use std::sync::Arc;
use std::time::Duration;

use armonik::events::{EventsEnum, Update};
use armonik::results::filter::{Condition, Field as FilterField, Status};
use armonik::{
    FilterStatusOperator, FilterString, FilterStringOperator, ResultStatus, SortDirection,
};
use futures::StreamExt;
use futures::stream::BoxStream;

use crate::pool::Pool;

/// A result of the session reached a final status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultUpdate {
    pub result_id: String,
    /// [`ResultStatus::Completed`] or [`ResultStatus::Aborted`].
    pub status: ResultStatus,
}

/// Source of the final statuses of a session's results.
pub trait ResultWatcher: Send + Sync {
    /// Streams the results of `session_id` whose name starts with
    /// `name_prefix` and that are completed or aborted, including those that
    /// already were when the stream starts. Every such result is reported at
    /// least once; duplicates are possible, so the caller filters. Transient
    /// errors are logged and retried: the stream never ends on its own.
    fn watch(
        &self,
        pool: Arc<Pool>,
        session_id: String,
        name_prefix: String,
    ) -> BoxStream<'static, ResultUpdate>;
}

/// The available [`ResultWatcher`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WatchMode {
    /// Subscribe to result status updates (GetEvents).
    Events,
    /// Periodically list the session's finished results (ListResults).
    Polling,
}

impl WatchMode {
    pub fn watcher(self, poll_interval: Duration) -> Box<dyn ResultWatcher> {
        match self {
            WatchMode::Events => Box::new(EventsWatcher),
            WatchMode::Polling => Box::new(PollingWatcher {
                interval: poll_interval,
            }),
        }
    }
}

/// Delay before retrying after an error.
const RETRY_DELAY: Duration = Duration::from_secs(1);
const PAGE_SIZE: i32 = 500;

/// Watches through the GetEvents stream. Each (re)subscription is followed
/// by a ListResults pass, which reports what finished before the stream was
/// established (before the first subscription, or while reconnecting).
pub struct EventsWatcher;

impl ResultWatcher for EventsWatcher {
    fn watch(
        &self,
        pool: Arc<Pool>,
        session_id: String,
        name_prefix: String,
    ) -> BoxStream<'static, ResultUpdate> {
        async_stream::stream! {
            loop {
                // The subscription holds its pooled connection while it lasts.
                let subscription = async {
                    let mut events = pool.events().await?;
                    let stream = events
                        .subscribe(
                            &session_id,
                            Vec::<Vec<armonik::tasks::filter::Field>>::new(),
                            [[name_filter(&name_prefix)]],
                            [EventsEnum::ResultStatusUpdate],
                        )
                        .await?;
                    Ok::<_, eyre::Report>((events, stream))
                };
                match subscription.await {
                    Ok((_connection, events)) => {
                        // Catch up only once subscribed, so nothing finishing
                        // in between can be missed.
                        match list_finished(&pool, &session_id, &name_prefix).await {
                            Ok(finished) => {
                                for update in finished {
                                    yield update;
                                }
                            }
                            Err(err) => {
                                tracing::warn!("could not list the finished results: {err:#}");
                                tokio::time::sleep(RETRY_DELAY).await;
                                continue;
                            }
                        }
                        let mut events = std::pin::pin!(events);
                        while let Some(event) = events.next().await {
                            match event {
                                Ok(event) => {
                                    if let Update::ResultStatusUpdate(update) = event.update
                                        && is_final(&update.status)
                                    {
                                        yield ResultUpdate {
                                            result_id: update.result_id,
                                            status: update.status,
                                        };
                                    }
                                }
                                Err(err) => {
                                    tracing::warn!("event stream failed: {}", crate::Chain(&err));
                                    break;
                                }
                            }
                        }
                        tracing::warn!("event stream closed, subscribing again");
                    }
                    Err(err) => tracing::warn!("could not subscribe to events: {err:#}"),
                }
                tokio::time::sleep(RETRY_DELAY).await;
            }
        }
        .boxed()
    }
}

/// Watches by listing the session's finished results every `interval`.
pub struct PollingWatcher {
    pub interval: Duration,
}

impl ResultWatcher for PollingWatcher {
    fn watch(
        &self,
        pool: Arc<Pool>,
        session_id: String,
        name_prefix: String,
    ) -> BoxStream<'static, ResultUpdate> {
        let interval = self.interval;
        async_stream::stream! {
            loop {
                match list_finished(&pool, &session_id, &name_prefix).await {
                    Ok(finished) => {
                        for update in finished {
                            yield update;
                        }
                    }
                    Err(err) => tracing::warn!("could not list the finished results: {err:#}"),
                }
                tokio::time::sleep(interval).await;
            }
        }
        .boxed()
    }
}

fn is_final(status: &ResultStatus) -> bool {
    matches!(status, ResultStatus::Completed | ResultStatus::Aborted)
}

fn name_filter(prefix: &str) -> FilterField {
    FilterField {
        field: armonik::results::Field::Name,
        condition: Condition::String(FilterString {
            value: prefix.to_owned(),
            operator: FilterStringOperator::StartsWith,
        }),
    }
}

/// All the completed or aborted results of the session named
/// `name_prefix...`, page by page (one pooled connection per page).
async fn list_finished(
    pool: &Pool,
    session_id: &str,
    name_prefix: &str,
) -> eyre::Result<Vec<ResultUpdate>> {
    let session = FilterField {
        field: armonik::results::Field::SessionId,
        condition: Condition::String(FilterString {
            value: session_id.to_owned(),
            operator: FilterStringOperator::Equal,
        }),
    };
    let filters = [ResultStatus::Completed, ResultStatus::Aborted].map(|status| {
        [
            session.clone(),
            name_filter(name_prefix),
            FilterField {
                field: armonik::results::Field::Status,
                condition: Condition::Status(Status {
                    value: status,
                    operator: FilterStatusOperator::Equal,
                }),
            },
        ]
    });
    let sort = armonik::results::Sort {
        field: armonik::results::Field::ResultId,
        direction: SortDirection::Asc,
    };

    let mut finished = Vec::new();
    for page in 0.. {
        let response = pool
            .results()
            .await?
            .list(filters.clone(), sort.clone(), page, PAGE_SIZE)
            .await?;
        let last = response.results.len() < PAGE_SIZE as usize;
        finished.extend(response.results.into_iter().map(|result| ResultUpdate {
            result_id: result.result_id,
            status: result.status,
        }));
        if last {
            break;
        }
    }
    Ok(finished)
}
