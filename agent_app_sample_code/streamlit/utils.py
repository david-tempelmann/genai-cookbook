"""Util functions for the Streamlit app."""
import io
import logging

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied, ResourceDoesNotExist, ResourceAlreadyExists
from databricks.sdk.service import jobs
from databricks.sdk.service.serving import EndpointStateReady
from databricks.sdk.service.workspace import ImportFormat

from agent_config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

browser_url = "e2-demo-field-eng.cloud.databricks.com"  # du.get_browser_hostname()


def create_uc_objects(w, UC_CATALOG, UC_SCHEMA, LLM_ENDPOINT):
    try:
        _ = w.catalogs.get(UC_CATALOG)
        logger.info(f"PASS: UC catalog `{UC_CATALOG}` exists")
    except NotFound as e:
        logger.info(f"`{UC_CATALOG}` does not exist, trying to create...")
        try:
            _ = w.catalogs.create(name=UC_CATALOG)
        except PermissionDenied as e:
            logger.info(
                f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create. Please provide an existing UC Catalog.")
            raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")

    # Create UC Schema if it does not exist, otherwise, raise an exception
    try:
        _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
        logger.info(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
    except NotFound as e:
        logger.info(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
        try:
            _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
            logger.info(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
        except PermissionDenied as e:
            logger.info(
                f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create. Please provide an existing UC Schema.")
            raise ValueError(f"Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

    task_type = "llm/v1/chat"
    try:
        llm_endpoint = w.serving_endpoints.get(name=LLM_ENDPOINT)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            logger.info(
                f"FAIL: Model serving endpoint {LLM_ENDPOINT} is not in a READY state. Please visit the status page to debug: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
        else:
            if llm_endpoint.task != task_type:
                logger.info(
                    f"FAIL: Model serving endpoint {LLM_ENDPOINT} is online & ready, but does not support task type /{task_type}. Details at: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
            else:
                logger.info(
                    f"PASS: Model serving endpoint {LLM_ENDPOINT} is online & ready and supports task type /{task_type}. Details at: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
    except ResourceDoesNotExist as e:
        logger.info(
            f"FAIL: Model serving endpoint {LLM_ENDPOINT} does not exist. Please create it at: https://{browser_url}/ml/endpoints/")


def create_config(w: WorkspaceClient, repo_path: str, UC_CATALOG, UC_SCHEMA, AGENT_NAME, LLM_ENDPOINT, TEMPERATURE):
    # Create and save agent config
    retriever_config = RetrieverToolConfig(
        vector_search_index=f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}_chunked_docs_index",
        vector_search_schema=RetrieverSchemaConfig(
            primary_key="chunk_id",
            chunk_text="content_chunked",
            document_uri="doc_uri",
            additional_metadata_columns=[],
        ),
        parameters=RetrieverParametersConfig(num_results=5, query_type="ann"),
        vector_search_threshold=0.1,
        chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",
        prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",
        tool_description_prompt="Search for documents that are relevant to a user's query about the [REPLACE WITH DESCRIPTION OF YOUR DOCS].",
    )

    llm_config = LLMConfig(
        llm_endpoint_name=LLM_ENDPOINT,
        llm_system_prompt_template=(
            """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the information from tools that are explictly specified to you.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
        ),
        llm_parameters=LLMParametersConfig(temperature=TEMPERATURE, max_tokens=1500),
    )

    agent_config = AgentConfig(
        retriever_tool=retriever_config,
        llm_config=llm_config,
        input_example={
            "messages": [
                {
                    "role": "user",
                    "content": "What is RAG?",
                },
            ]
        },
    )

    # Save the agent config
    w.workspace.upload(
        path=repo_path + "/agent_app_sample_code/agents/generated_configs/agent.yaml",
        content=io.BytesIO(yaml.dump(agent_config.dict()).encode("utf-8")),
        format=ImportFormat.AUTO,
        overwrite=True
    )


def submit_jobs(w: WorkspaceClient, cluster_id: str, UC_CATALOG, UC_SCHEMA, UC_MODEL_NAME, AGENT_NAME,
                POC_DATA_PIPELINE_RUN_NAME, POC_CHAIN_RUN_NAME, MLFLOW_EXPERIMENT_NAME, notebook_base_path):
    base_parameters = {
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        "UC_CATALOG": UC_CATALOG,
        "UC_SCHEMA": UC_SCHEMA,
        "UC_MODEL_NAME": UC_MODEL_NAME,
        "AGENT_NAME": AGENT_NAME,
        "POC_DATA_PIPELINE_RUN_NAME": POC_DATA_PIPELINE_RUN_NAME,
        "POC_CHAIN_RUN_NAME": POC_CHAIN_RUN_NAME,
    }

    logger.info(f"Base parameters: {base_parameters}")

    logger.info("Create or get experiment")
    try:
        experiment = w.experiments.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
    except ResourceAlreadyExists as e:
        experiment = w.experiments.get_by_name(MLFLOW_EXPERIMENT_NAME).experiment

    experiment_id = experiment.experiment_id
    logger.info(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' ID: {experiment_id}")

    logger.info("Define data pipeline task")
    data_pipeline_job = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{notebook_base_path}/02_data_pipeline",
            base_parameters=base_parameters,
        ),
        task_key="02_data_pipeline",
    )

    logger.info("Define agent deploy task")
    agent_job = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=f"{notebook_base_path}/03_agent_proof_of_concept",
            base_parameters=base_parameters,
        ),
        task_key="03_agent_poc",
        depends_on=[data_pipeline_job],
    )

    logger.info("Submit jobs")
    job_run = w.jobs.submit(
        run_name="test_tuning",
        tasks=[data_pipeline_job, agent_job],
    )


    logger.info(f"Experiment with name {MLFLOW_EXPERIMENT_NAME} has ID: {experiment_id}")
    mlflow_exp_link = f"https://{browser_url}/ml/experiments/{experiment_id}"
    job_link = w.jobs.get_run(job_run.run_id).run_page_url

    st.info(f"Experiment link: {mlflow_exp_link}")
    st.info(f"Job Run link: {job_link}")

    return job_run
