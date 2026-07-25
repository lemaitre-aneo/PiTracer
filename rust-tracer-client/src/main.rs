use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Clone, Default, Parser)]
pub struct Cli {
    /// ArmoniK control-plane gRPC endpoint
    #[arg(short, long)]
    pub endpoint: String,

    /// Number of gRPC channels to open
    #[arg(short, long, default_value = "1")]
    pub channels: usize,

    /// Number of concurrent request streams per channel
    #[arg(short, long, default_value = "1")]
    pub streams: usize,

    /// Seconds to wait before exiting
    #[arg(short, long, default_value = "0")]
    pub wait: f64,
}

#[tokio::main]
async fn main() -> Result<(), eyre::Report> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::NEW
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        ))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let mut config = armonik::ClientConfig::default();
    config.endpoint = cli.endpoint.parse().unwrap();

    let client = armonik::Client::with_config(config).await?;
    Ok(())
}
