//! Command-line surface of the connection to the ArmoniK control plane: one
//! flag per `armonik-transport` option, each falling back to its
//! `GrpcClient__*` environment variable (the ArmoniK convention), handed to
//! `armonik-transport` as given, which validates them; plus the connection
//! pool's settings (`pool.rs`).

use armonik::transport::{ClientConfig, ClientConfigArgs};

#[derive(Debug, Clone, clap::Args)]
#[command(next_help_heading = "Connection")]
pub struct Transport {
    /// ArmoniK control-plane endpoint; `host:port` means `http://host:port`
    #[arg(short, long, alias = "server-url", env = "GrpcClient__Endpoint")]
    pub endpoint: String,

    /// Client certificate (PEM file) for mTLS
    #[arg(long, env = "GrpcClient__CertPem", requires = "key_pem")]
    pub cert_pem: Option<String>,

    /// Client private key (PEM file) for mTLS
    #[arg(long, env = "GrpcClient__KeyPem", requires = "cert_pem")]
    pub key_pem: Option<String>,

    /// Certificate authority (PEM file) to verify the server with
    #[arg(long, env = "GrpcClient__CaCert")]
    pub ca_cert: Option<String>,

    /// Skip the verification of the server certificate
    #[arg(long, env = "GrpcClient__AllowUnsafeConnection")]
    pub allow_unsafe_connection: bool,

    /// Server name to verify the certificate against, instead of the
    /// endpoint's
    #[arg(long, env = "GrpcClient__OverrideTargetName")]
    pub override_target_name: Option<String>,

    /// Timeout for establishing the connection (e.g. `30s`) [default: none]
    #[arg(long, env = "GrpcClient__ConnectTimeout")]
    pub connect_timeout: Option<String>,

    /// Timeout of every request, the GetEvents stream included (e.g. `1m`)
    /// [default: none]
    #[arg(long, env = "GrpcClient__Timeout")]
    pub timeout: Option<String>,

    /// Request rate limit, as `count/duration` (e.g. `100/1s`) [default: none]
    #[arg(long, env = "GrpcClient__RateLimit")]
    pub rate_limit: Option<String>,

    /// TCP keepalive duration (e.g. `30s`) [default: none]
    #[arg(long, env = "GrpcClient__TcpKeepalive")]
    pub tcp_keepalive: Option<String>,

    /// Interval between TCP keepalive probes (e.g. `5s`) [default: OS]
    #[arg(long, env = "GrpcClient__TcpKeepaliveInterval")]
    pub tcp_keepalive_interval: Option<String>,

    /// Number of TCP keepalive retries [default: OS]
    #[arg(long, env = "GrpcClient__TcpKeepaliveRetries")]
    pub tcp_keepalive_retries: Option<String>,

    /// Enable Nagle's algorithm (disables TCP_NODELAY)
    #[arg(long, env = "GrpcClient__TcpNagleAlgorithm")]
    pub tcp_nagle_algorithm: bool,

    /// HTTP/2 PING interval (e.g. `20s`); keeps an idle GetEvents stream
    /// alive through proxies [default: none]
    #[arg(long, env = "GrpcClient__Http2KeepAliveInterval")]
    pub http2_keep_alive_interval: Option<String>,

    /// HTTP/2 PING timeout (e.g. `10s`) [default: none]
    #[arg(long, env = "GrpcClient__Http2KeepAliveTimeout")]
    pub http2_keep_alive_timeout: Option<String>,

    /// Send HTTP/2 PINGs even without a request in flight
    #[arg(long, env = "GrpcClient__Http2KeepAliveWhileIdle")]
    pub http2_keep_alive_while_idle: bool,

    /// HTTP/2 maximum header list size, in bytes [default: none]
    #[arg(long, env = "GrpcClient__Http2MaxHeaderListSize")]
    pub http2_max_header_list_size: Option<String>,

    /// User-Agent header of every request [default: rust-tracer-client/<version>]
    #[arg(long, env = "GrpcClient__UserAgent")]
    pub user_agent: Option<String>,

    /// Maximum number of connections to the control plane [default: 4 x
    /// tokio workers; 0: unlimited]
    #[arg(long)]
    pub pool_size: Option<usize>,

    /// Requests sent on a connection before it is replaced, to stay below
    /// the ingress's per-connection cap (nginx closes connections after
    /// 1000 requests by default) [default: 800; 0: never]
    #[arg(long)]
    pub requests_per_connection: Option<usize>,

    /// Let a connection carry several requests at once
    #[arg(long)]
    pub multiplex: bool,
}

impl Transport {
    pub fn pool_config(&self) -> crate::pool::PoolConfig {
        crate::pool::PoolConfig {
            pool_size: self.pool_size,
            requests_per_connection: self.requests_per_connection,
            multiplex: self.multiplex,
        }
    }

    pub fn config(&self) -> Result<ClientConfig, armonik::transport::ConfigError> {
        let mut args = ClientConfigArgs::default();
        args.endpoint = if self.endpoint.contains("://") {
            self.endpoint.clone()
        } else {
            format!("http://{}", self.endpoint)
        };
        let set = |field: &mut String, value: &Option<String>| {
            if let Some(value) = value {
                field.clone_from(value);
            }
        };
        set(&mut args.cert_pem, &self.cert_pem);
        set(&mut args.key_pem, &self.key_pem);
        set(&mut args.ca_cert, &self.ca_cert);
        args.allow_unsafe_connection = self.allow_unsafe_connection;
        set(&mut args.override_target_name, &self.override_target_name);
        set(&mut args.connect_timeout, &self.connect_timeout);
        set(&mut args.timeout, &self.timeout);
        set(&mut args.rate_limit, &self.rate_limit);
        set(&mut args.tcp_keepalive, &self.tcp_keepalive);
        set(
            &mut args.tcp_keepalive_interval,
            &self.tcp_keepalive_interval,
        );
        set(&mut args.tcp_keepalive_retries, &self.tcp_keepalive_retries);
        args.tcp_nagle_algorithm = self.tcp_nagle_algorithm;
        set(
            &mut args.http2_keep_alive_interval,
            &self.http2_keep_alive_interval,
        );
        set(
            &mut args.http2_keep_alive_timeout,
            &self.http2_keep_alive_timeout,
        );
        args.http2_keep_alive_while_idle = self.http2_keep_alive_while_idle;
        set(
            &mut args.http2_max_header_list_size,
            &self.http2_max_header_list_size,
        );
        args.user_agent = self
            .user_agent
            .clone()
            .unwrap_or_else(|| concat!("rust-tracer-client/", env!("CARGO_PKG_VERSION")).into());
        ClientConfig::from_config_args(args)
    }
}
