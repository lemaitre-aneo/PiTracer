use std::sync::Arc;

use armonik::reexports::{tokio_stream::wrappers::UnixListenerStream, tonic};
use serde::{Deserialize, Serialize};
use tokio::net::UnixListener;
use tracing as _;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

mod worker;

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// ArmoniK deployments spell the values in lowercase.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
enum SocketType {
    #[serde(alias = "tcp")]
    Tcp,
    #[default]
    #[serde(alias = "unixdomainsocket")]
    UnixDomainSocket,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct GrpcChannel {
    /// A socket path, or for TCP a URL (`http://localhost:10667`, as ArmoniK
    /// deployments set it) or a bare socket address.
    address: String,
    #[serde(default)]
    #[serde(alias = "sockettype")]
    socket_type: SocketType,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Config {
    #[serde(alias = "agentchannel")]
    agent_channel: GrpcChannel,
    #[serde(alias = "workerchannel")]
    worker_channel: GrpcChannel,
    #[serde(default)]
    #[serde(alias = "abortafter")]
    abort_after: std::time::Duration,
}

#[tokio::main]
async fn main() -> Result<(), eyre::Report> {
    let env_filter = tracing_subscriber::EnvFilter::from_default_env();
    tracing_subscriber::registry()
        // Print only events matching the env filter, but record (without
        // printing) all spans regardless of their level, so events keep
        // their span context.
        .with(
            tracing_subscriber::fmt::layer()
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NONE)
                .with_filter(tracing_subscriber::filter::dynamic_filter_fn({
                    let env_filter = env_filter.clone();
                    move |metadata, context| {
                        metadata.is_span() || env_filter.enabled(metadata, context.clone())
                    }
                })),
        )
        // Print spans matching the env filter, but not events to avoid duplication
        .with(
            tracing_subscriber::fmt::layer()
                .with_span_events(
                    tracing_subscriber::fmt::format::FmtSpan::NEW
                        | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
                )
                .with_filter(tracing_subscriber::filter::dynamic_filter_fn({
                    let env_filter = env_filter.clone();
                    move |metadata, context| {
                        metadata.is_span() && env_filter.enabled(metadata, context.clone())
                    }
                })),
        )
        .init();

    let conf = config::Config::builder()
        .add_source(
            config::Environment::with_prefix("ComputePlane")
                .convert_case(config::Case::Snake)
                .separator("__"),
        )
        .set_default("pouet", "plop")?
        .build()?;

    tracing::trace!("{conf:?}");
    let conf: Config = conf.try_deserialize()?;

    tracing::trace!("{conf:?}");

    let worker = Arc::new(worker::Worker::new(conf.agent_channel)?);

    let router = tonic::transport::Server::builder().add_service(
        armonik::api::v3::worker::worker_server::WorkerServer::from_arc(worker.clone()),
    );

    let mut service_future = tokio::spawn(async move {
        match conf.worker_channel.socket_type {
            SocketType::Tcp => {
                router
                    .serve(tcp_address(&conf.worker_channel.address).await?)
                    .await?
            }
            SocketType::UnixDomainSocket => {
                // A socket file left over by a previous run would make the
                // bind fail.
                match std::fs::remove_file(&conf.worker_channel.address) {
                    Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(err)?,
                    _ => (),
                }
                let uds = UnixListener::bind(conf.worker_channel.address)?;
                let uds_stream = UnixListenerStream::new(uds);

                router.serve_with_incoming(uds_stream).await?;
            }
        }
        Ok::<_, eyre::Report>(())
    });

    tracing::info!("Worker running");

    tokio::select! {
        output = &mut service_future => {
            match output {
                Ok(Ok(())) => (),
                Ok(Err(err)) => {
                    tracing::error!("Service had an error: {err:?}");
                }
                Err(err) => {
                    tracing::error!("Service future had an error: {err:?}");
                }
            }
        }
        _ = wait_terminate() => {
            tracing::info!("Worker stopping");
            service_future.abort();
            _ = service_future.await;
        }
    }

    tracing::info!("Worker stopped");

    Ok(())
}

/// Socket address to listen on for a TCP channel address.
async fn tcp_address(address: &str) -> Result<std::net::SocketAddr, eyre::Report> {
    if let Ok(address) = address.parse() {
        return Ok(address);
    }
    let uri: tonic::transport::Uri = address.parse()?;
    let host = uri
        .host()
        .ok_or_else(|| eyre::eyre!("no host in {address}"))?;
    let port = uri
        .port_u16()
        .ok_or_else(|| eyre::eyre!("no port in {address}"))?;
    tokio::net::lookup_host((host, port))
        .await?
        .next()
        .ok_or_else(|| eyre::eyre!("{address} resolves to no address"))
}

/// Wait for termination signal (either SIGINT or SIGTERM)
#[cfg(unix)]
async fn wait_terminate() {
    use futures::{StreamExt, stream::FuturesUnordered};
    use tokio::signal::unix::{SignalKind, signal};
    let mut signals = Vec::new();

    // Register signal handlers
    for sig in [SignalKind::terminate(), SignalKind::interrupt()] {
        match signal(sig) {
            Ok(sig) => signals.push(sig),
            Err(err) => tracing::error!("Could not register signal handler: {err}"),
        }
    }

    // Wait for the first signal to trigger
    let mut signals = signals
        .iter_mut()
        .map(|sig| sig.recv())
        .collect::<FuturesUnordered<_>>();

    loop {
        match signals.next().await {
            // a signal fired: stop waiting
            Some(Some(())) => break,
            // this handler was shut down: keep waiting on the others
            Some(None) => (),
            // no signal handler could be registered: wait indefinitely
            None => futures::future::pending::<()>().await,
        }
    }
}

#[cfg(windows)]
macro_rules! win_signal {
    ($($sig:ident),*$(,)?) => {
        $(
            let $sig = async {
                match tokio::signal::windows::$sig() {
                    Ok(mut $sig) => {
                        if $sig.recv().await.is_some() {
                            return;
                        }
                    }
                    Err(err) => tracing::error!(
                        "Could not register signal handler for {}: {err}",
                        stringify!($sig),
                    ),
                }
                futures::future::pending::<()>().await;
            };
        )*
        tokio::select! {
            $(
                _ = $sig => {}
            )*
        }
    }
}

/// Wait for a termination signal (Ctrl+C, close, logoff, or shutdown).
#[cfg(windows)]
async fn wait_terminate() {
    win_signal!(ctrl_c, ctrl_close, ctrl_logoff, ctrl_shutdown);
}
