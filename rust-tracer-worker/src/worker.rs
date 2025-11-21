use std::sync::Arc;

use armonik::{reexports::tonic, server, worker};

pub struct Worker {
    agent: armonik::client::Agent<armonik::Client>,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker").finish()
    }
}

impl Worker {
    pub async fn new(channel_config: crate::GrpcChannel) -> Result<Self, eyre::Report> {
        let client = match channel_config.socket_type {
            crate::SocketType::Tcp => {
                let mut config = armonik::ClientConfig::default();
                config.endpoint = channel_config.address.parse()?;
                armonik::Client::with_config(config).await?
            }
            crate::SocketType::UnixDomainSocket => todo!(),
        };

        Ok(Self {
            agent: client.into_agent(),
        })
    }
}

impl server::WorkerService for Worker {
    async fn health_check(
        self: Arc<Self>,
        _request: worker::health_check::Request,
        _context: server::RequestContext,
    ) -> std::result::Result<worker::health_check::Response, tonic::Status> {
        Ok(worker::health_check::Response::Serving)
    }

    #[tracing::instrument()]
    async fn process(
        self: Arc<Self>,
        _request: worker::process::Request,
        _context: server::RequestContext,
    ) -> std::result::Result<worker::process::Response, tonic::Status> {
        Err(tonic::Status::unimplemented("Not Implemented"))
    }
}
