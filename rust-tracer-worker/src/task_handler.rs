//! Task-side view of an ArmoniK `Process` call, modeled on the C# API's
//! `ITaskHandler` (`ArmoniK.Api.Worker/Worker/TaskHandler.cs`): it unifies
//! the task's inputs (payload and data dependencies, exchanged as files in
//! the data folder shared with the agent) and the agent calls a task may
//! make (send its results, create new results, submit new tasks).
//!
//! Nothing here is specific to the ray tracer.

use std::path::PathBuf;

use armonik::agent::{ResultMetaData, submit_tasks};
use armonik::client::Agent;
use armonik::reexports::tonic::transport::Channel;
use armonik::{Configuration, TaskOptions, worker};
use eyre::{Context, ensure};

pub struct TaskHandler {
    agent: Agent<Channel>,
    /// Identifies the task in every agent request.
    token: String,
    session_id: String,
    task_id: String,
    task_options: TaskOptions,
    payload: Vec<u8>,
    data_dependencies: Vec<String>,
    expected_results: Vec<String>,
    configuration: Configuration,
    data_folder: PathBuf,
}

impl std::fmt::Debug for TaskHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskHandler")
            .field("session_id", &self.session_id)
            .field("task_id", &self.task_id)
            .finish_non_exhaustive()
    }
}

impl TaskHandler {
    /// Reads the task payload from the data folder.
    pub async fn new(
        request: worker::process::Request,
        agent: Agent<Channel>,
    ) -> eyre::Result<Self> {
        let data_folder = PathBuf::from(request.data_folder);
        let payload = tokio::fs::read(data_folder.join(&request.payload_id))
            .await
            .wrap_err_with(|| format!("could not read payload {}", request.payload_id))?;

        Ok(Self {
            agent,
            token: request.communication_token,
            session_id: request.session_id,
            task_id: request.task_id,
            task_options: request.task_options,
            payload,
            data_dependencies: request.data_dependencies,
            expected_results: request.expected_output_keys,
            configuration: request.configuration,
            data_folder,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Options the task was submitted with.
    pub fn task_options(&self) -> &TaskOptions {
        &self.task_options
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Ids of the results the task depends on, readable with
    /// [`TaskHandler::data_dependency`].
    pub fn data_dependencies(&self) -> &[String] {
        &self.data_dependencies
    }

    /// Ids of the results the task must either send or delegate to a
    /// subtask.
    pub fn expected_results(&self) -> &[String] {
        &self.expected_results
    }

    pub fn configuration(&self) -> &Configuration {
        &self.configuration
    }

    /// Reads one of the task's data dependencies.
    pub async fn data_dependency(&self, result_id: &str) -> eyre::Result<Vec<u8>> {
        ensure!(
            self.data_dependencies.iter().any(|id| id == result_id),
            "{result_id} is not a data dependency of the task"
        );
        tokio::fs::read(self.data_folder.join(result_id))
            .await
            .wrap_err_with(|| format!("could not read data dependency {result_id}"))
    }

    /// Writes the data of a result owned by the task and notifies the agent
    /// it is available.
    pub async fn send_result(
        &mut self,
        result_id: &str,
        data: impl AsRef<[u8]>,
    ) -> eyre::Result<()> {
        tokio::fs::write(self.data_folder.join(result_id), data)
            .await
            .wrap_err_with(|| format!("could not write result {result_id}"))?;
        self.agent
            .notify_result_data(&self.token, &self.session_id, [result_id])
            .await
            .wrap_err_with(|| format!("could not notify result {result_id}"))?;
        Ok(())
    }

    /// Creates results without data (to be produced by this task or one of
    /// its subtasks), one per name.
    pub async fn create_results_metadata(
        &mut self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> eyre::Result<Vec<ResultMetaData>> {
        self.agent
            .create_results_metadata(&self.token, &self.session_id, names)
            .await
            .wrap_err("could not create results metadata")
    }

    /// Creates results with their data (e.g. the payloads of subtasks).
    pub async fn create_results(
        &mut self,
        results: impl IntoIterator<Item = (impl Into<String>, impl Into<Vec<u8>>)>,
    ) -> eyre::Result<Vec<ResultMetaData>> {
        self.agent
            .create_results(&self.token, &self.session_id, results)
            .await
            .wrap_err("could not create results")
    }

    /// Submits tasks whose payloads and data dependencies are existing
    /// results. `task_options` defaults those of the submitted tasks that
    /// carry none; without it, the session's options apply.
    pub async fn submit_tasks(
        &mut self,
        task_options: Option<TaskOptions>,
        tasks: impl IntoIterator<Item = submit_tasks::RequestItem>,
    ) -> eyre::Result<Vec<submit_tasks::ResponseItem>> {
        self.agent
            .submit_tasks(&self.token, &self.session_id, task_options, tasks)
            .await
            .wrap_err("could not submit tasks")
    }
}
