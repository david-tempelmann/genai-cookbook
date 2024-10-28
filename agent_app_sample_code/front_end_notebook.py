# Databricks notebook source
# Your original code goes here, using the input values
user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]
# Get the current notebook's context
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# Get the current cluster ID
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"Current cluster ID: {cluster_id}")

# COMMAND ----------

# MAGIC %run ./agents/agent_config

# COMMAND ----------

import ipywidgets as widgets
from IPython.display import display, HTML
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from mlflow.utils import databricks_utils as du
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os
from databricks.sdk.service.compute import DataSecurityMode
from pyspark.sql import SparkSession
from databricks.sdk.service import jobs
from pathlib import Path
from mlflow.utils.databricks_utils import get_notebook_path
import mlflow
from mlflow.tracking import MlflowClient

# Define a common style and layout
style = {'description_width': 'initial'}
layout = widgets.Layout(width='500px')

# Create input widgets
agent_name_input = widgets.Text(description='AGENT_NAME:', style=style, layout=layout)
uc_catalog_input = widgets.Text(description='UC_CATALOG:', style=style, layout=layout)
uc_schema_input = widgets.Text(description='UC_SCHEMA:', style=style, layout=layout)
llm_endpoint_input = widgets.Text(description='LLM Endpoint:', style=style, layout=layout)
temperature_input = widgets.FloatSlider(
    description='Temperature:',
    min=0, max=1, step=0.01, value=0.01,
    style=style,
    layout=widgets.Layout(width='500px')
)

# Create output widget
output = widgets.Output()

# Create button widget
run_button = widgets.Button(description='Run', layout=widgets.Layout(width='100px'))

# Define button click handler
def on_button_click(b):
    with output:
        output.clear_output()
        
        # Get input values
        AGENT_NAME = agent_name_input.value
        UC_CATALOG = uc_catalog_input.value
        UC_SCHEMA = uc_schema_input.value

        LLM_ENDPOINT = llm_endpoint_input.value
        TEMPERATURE = temperature_input.value

        UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}"
        EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_evaluation_set`"
        MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{AGENT_NAME}"
        POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
        POC_CHAIN_RUN_NAME = "agent_poc"

        print("--user info--")
        print(f"user_name {user_name}")

        print("--agent--")
        print(f"AGENT_NAME {AGENT_NAME}")
        print(f"UC_CATALOG {UC_CATALOG}")
        print(f"UC_SCHEMA {UC_SCHEMA}")
        print(f"UC_MODEL_NAME {UC_MODEL_NAME}")

        print()
        print("--evaluation config--")
        print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
        print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
        print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
        print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")

        # Create UC Catalog if it does not exist, otherwise, raise an exception
        w = WorkspaceClient(host=host, token=token)
        mlclient = MlflowClient()
        try:
            _ = w.catalogs.get(UC_CATALOG)
            print(f"PASS: UC catalog `{UC_CATALOG}` exists")
        except NotFound as e:
            print(f"`{UC_CATALOG}` does not exist, trying to create...")
            try:
                _ = w.catalogs.create(name=UC_CATALOG)
            except PermissionDenied as e:
                print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create. Please provide an existing UC Catalog.")
                raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")

        # Create UC Schema if it does not exist, otherwise, raise an exception
        try:
            _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
            print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
        except NotFound as e:
            print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
            try:
                _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
                print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
            except PermissionDenied as e:
                print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create. Please provide an existing UC Schema.")
                raise ValueError(f"Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

        browser_url = du.get_browser_hostname()
        task_type = "llm/v1/chat"
        try:
            llm_endpoint = w.serving_endpoints.get(name=LLM_ENDPOINT)
            if llm_endpoint.state.ready != EndpointStateReady.READY:
                print(f"FAIL: Model serving endpoint {LLM_ENDPOINT} is not in a READY state. Please visit the status page to debug: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
            else:
                if llm_endpoint.task != task_type:
                    print(f"FAIL: Model serving endpoint {LLM_ENDPOINT} is online & ready, but does not support task type /{task_type}. Details at: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
                else:
                    print(f"PASS: Model serving endpoint {LLM_ENDPOINT} is online & ready and supports task type /{task_type}. Details at: https://{browser_url}/ml/endpoints/{LLM_ENDPOINT}")
        except ResourceDoesNotExist as e:
            print(f"FAIL: Model serving endpoint {LLM_ENDPOINT} does not exist. Please create it at: https://{browser_url}/ml/endpoints/")

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
        save_agent_config(agent_config.dict(), './agents/generated_configs/agent.yaml')
        print("Agent config saved successfully.")
        print("-------------------------------------")
        print("Trigger agent deployment run ...")

        def notebook_path(relative_path: str):
            return str(Path(get_notebook_path()).parent.joinpath(relative_path).resolve())

        base_parameters = {
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        "UC_CATALOG": UC_CATALOG,
        "UC_SCHEMA": UC_SCHEMA,
        "UC_MODEL_NAME": UC_MODEL_NAME,
        "AGENT_NAME": AGENT_NAME,
        "POC_DATA_PIPELINE_RUN_NAME": POC_DATA_PIPELINE_RUN_NAME,
        "POC_CHAIN_RUN_NAME": POC_CHAIN_RUN_NAME,
        }

        data_pipeline_job = jobs.SubmitTask(
                existing_cluster_id=cluster_id,
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path("./02_data_pipeline"),
                    base_parameters=base_parameters,
                ),
                task_key="02_data_pipeline",
            )

        agent_job = jobs.SubmitTask(
                existing_cluster_id=cluster_id,
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path("./03_agent_proof_of_concept"),
                    base_parameters=base_parameters,
                ),
                task_key="03_agent_poc",
                depends_on=[data_pipeline_job],
            )
        
        job_run = w.jobs.submit(
            run_name="test_tuning",
            tasks=[data_pipeline_job, agent_job],
        )

        mlflow_exp_link = f"https://{browser_url}/ml/experiments/{mlclient.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id}"
        job_link = w.jobs.get_run(job_run.run_id).run_page_url

        print(f"Experiment link: {mlflow_exp_link}")
        print(f"Job Run link: {job_link}")

# Attach the handler to the button
run_button.on_click(on_button_click)

# Display the widgets
display(agent_name_input, uc_catalog_input, uc_schema_input,llm_endpoint_input,temperature_input, run_button, output)

# COMMAND ----------

# my_agent_app_test
# david_tempelmann
# genai_cookbook
# databricks-meta-llama-3-1-70b-instruct

# COMMAND ----------

# DBTITLE 1,Temporary cell
AGENT_NAME = agent_name_input.value
UC_CATALOG = uc_catalog_input.value
UC_SCHEMA = uc_schema_input.value

LLM_ENDPOINT = llm_endpoint_input.value
TEMPERATURE = temperature_input.value

UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}"
EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_evaluation_set`"
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{AGENT_NAME}"
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
POC_CHAIN_RUN_NAME = "agent_poc"

w = WorkspaceClient()


# COMMAND ----------

POC_DATA_PIPELINE_RUN_NAME
# POC_CHAIN_RUN_NAME

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
mlclient = MlflowClient()

browser_url = du.get_browser_hostname()

mlflow_exp_link = f"https://{browser_url}/ml/experiments/{mlclient.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id}"
print(mlflow_exp_link)


# COMMAND ----------


browser_url

# COMMAND ----------



def notebook_path(relative_path: str):
    return str(Path(get_notebook_path()).parent.joinpath(relative_path).resolve())

base_parameters = {
  "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
  "UC_CATALOG": UC_CATALOG,
  "UC_SCHEMA": UC_SCHEMA,
  "UC_MODEL_NAME": UC_MODEL_NAME,
  "AGENT_NAME": AGENT_NAME,
  "POC_DATA_PIPELINE_RUN_NAME": POC_DATA_PIPELINE_RUN_NAME,
  "POC_CHAIN_RUN_NAME": POC_CHAIN_RUN_NAME,
}

data_pipeline_job = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path("./02_data_pipeline"),
            base_parameters=base_parameters,
        ),
        task_key="02_data_pipeline",
    )

agent_job = jobs.SubmitTask(
        existing_cluster_id=cluster_id,
        notebook_task=jobs.NotebookTask(
            notebook_path=notebook_path("./03_agent_proof_of_concept"),
            base_parameters=base_parameters,
        ),
        task_key="03_agent_poc",
        depends_on=[data_pipeline_job],
    )

# COMMAND ----------

job_run = w.jobs.submit(
    run_name="test_tuning",
    tasks=[data_pipeline_job, agent_job],
)

#print(f"Job ID: {job_run.job_id}")

# COMMAND ----------

job_run.response.run_id

# COMMAND ----------

w.jobs.get_run(job_run.run_id).run_page_url

# COMMAND ----------

print(f"https://{browser_url}/jobs/{w.jobs.get_run(job_run.run_id).job_id}/runs/{job_run.run_id}")

# COMMAND ----------

job_run.result()

# COMMAND ----------


