import logging

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import WorkspaceObjectAccessControlRequest, WorkspaceObjectPermissionLevel
from mlflow import MlflowClient

from utils import create_uc_objects, create_config, submit_jobs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the WorkspaceClient and MlflowClient
w = WorkspaceClient()
mlclient = MlflowClient()

# get current user
current_user = w.current_user.me().user_name

# Set some parameters
repo_path = f"/Users/{current_user}/genai_cookbook_ui"
github_url = "https://github.com/david-tempelmann/genai-cookbook.git"
cluster_id = "0608-094354-z4kgacep"
repo_branch = "explore_jobs"

st.title("Run & Deploy GenAI Cookbook")

logger.info("Streamlit app started")

# Initialize session state
if 'repo_created' not in st.session_state:
    st.session_state.repo_created = False

# Some input fields
user_to_grant_access = st.text_input("Enter username to grant view access", value="david.tempelmann@databricks.com")
AGENT_NAME = st.text_input("Enter agent name", value="genai_cookbook_test_dt")
UC_CATALOG = st.text_input("Enter UC Catalog", value="david_tempelmann")
UC_SCHEMA = st.text_input("Enter UC Schema", value="genai_cookbook")
LLM_ENDPOINT = st.text_input("Enter LLM Endpoint", value="databricks-meta-llama-3-70b-instructor")
TEMPERATURE = st.slider("Select temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Some derived parameters
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}"
EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_evaluation_set`"
MLFLOW_EXPERIMENT_NAME = f"/Users/{current_user}/{AGENT_NAME}"
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
POC_CHAIN_RUN_NAME = "agent_poc"


def repo_exists(path):
    try:
        w.workspace.get_status(path)
        return True
    except Exception:
        return False


if not st.session_state.repo_created:
    if st.button("Run & Deploy"):

        logger.info("Run & Deploy button clicked")

        try:
            with st.spinner("Set up resources"):
                # Check and create UC objects
                logger.info("Checking/creating UC objects")
                create_uc_objects(w, UC_CATALOG, UC_SCHEMA, LLM_ENDPOINT)

                if repo_exists(f"{repo_path}"):
                    logger.warning(f"Repository already exists at path: {repo_path}")
                    st.warning(f"Repository already exists at path: {repo_path}")
                else:
                    # Create the repo
                    logger.info(f"Attempting to create repo at path: {repo_path}")
                    repo = w.repos.create(
                        url=github_url,
                        provider="github",
                        path=repo_path
                    )

                    w.repos.update(
                        repo_id=repo.id,
                        branch=repo_branch
                    )

                    logger.info(f"Repo created successfully with ID: {repo.id}")
                    st.success(f"Repo created successfully with ID: {repo.id}")

                    # Set up permissions request
                    logger.info(f"Setting up permissions for user: {user_to_grant_access}")
                    access_control_list = [
                        WorkspaceObjectAccessControlRequest(
                            user_name=user_to_grant_access,
                            permission_level=WorkspaceObjectPermissionLevel.CAN_EDIT
                        )
                    ]

                    # Apply permissions
                    w.workspace.set_permissions(
                        workspace_object_type="repos",
                        workspace_object_id=str(repo.id),
                        access_control_list=access_control_list
                    )

                    logger.info(f"Edit permissions granted to {user_to_grant_access}")
                    st.success(f"Edit permissions granted to {user_to_grant_access}")

                logger.info("Create a folder for the generated config if not exists")
                configs_folder = f"{repo_path}/agent_app_sample_code/agents/generated_configs"
                w.workspace.mkdirs(configs_folder)
                logger.info(f"Folder {configs_folder} created successfully")

                # Create the agent config
                logger.info("Creating agent config")
                create_config(
                    w, repo_path, UC_CATALOG, UC_SCHEMA, AGENT_NAME, LLM_ENDPOINT, TEMPERATURE)
                logger.info("Agent config created successfully")
                st.success("Agent config created successfully")

            with st.spinner("Run data prep and deploy agent"):

                notebook_base_path = f"{repo_path}/agent_app_sample_code"
                logger.info(f"Submitting jobs using notebook base path: {notebook_base_path}")
                job_run = submit_jobs(w, cluster_id, UC_CATALOG, UC_SCHEMA, UC_MODEL_NAME, AGENT_NAME,
                                      POC_DATA_PIPELINE_RUN_NAME, POC_CHAIN_RUN_NAME, MLFLOW_EXPERIMENT_NAME,
                                      notebook_base_path)
                logger.info("Jobs submitted successfully")

                job_result = job_run.result()

            # Set the flag to indicate repo has been created
            st.session_state.repo_created = True
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            logger.error(error_message)
            st.error(error_message)

# # Follow-up dialogue for temperature input
# if st.session_state.repo_created:
#     st.header("Additional Configuration")
#     temperature = st.slider("Select temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
#     if st.button("Save Temperature"):
#         logger.info(f"Temperature set to: {temperature}")
#         st.success(f"Temperature set to: {temperature}")
#         # Here you can add code to use or store the temperature value as needed

logger.info("Streamlit app execution completed")
