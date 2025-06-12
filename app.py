import gradio as gr
import pandas as pd
import sqlite3
from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import PreprocessingAgent
import os

# Global variables (will now primarily use gr.State for passing between functions)
# These global vars are mostly for initial default values or if you need persistent access outside Gradio's state
# For Gradio's internal flow, gr.State is better
global_dataset = None
global_reason = None
global_preprocessed_dataset = None
global_preprocess_reason = None
global_metadata = None

def load_csv_file(file):
    try:
        df = pd.read_csv(file.name)
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        return df, f"Loaded dataset from {file.name}."
    except Exception as e:
        return None, f"Failed to load CSV: {str(e)}"

def load_sqlite_db(db_file, table_name):
    try:
        conn = sqlite3.connect(db_file.name)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        if df.empty:
            raise ValueError("Database table is empty.")
        return df, f"Loaded dataset from SQLite database {db_file.name}, table {table_name}."
    except Exception as e:
        return None, f"Failed to load SQLite database: {str(e)}"

# --- New Function for Dataset Acquisition Step ---
def process_dataset_acquisition(dataset_source, prompt, num_samples, features, csv_file, db_file, table_name, current_dataset_state, current_metadata_state):
    """
    Handles loading or generating the dataset and initial metadata.
    Returns the dataset, metadata, and a message.
    """
    
    local_dataset = None
    local_reason = None

    if dataset_source == "Generate Synthetic Dataset":
        dataset_requirements = f"num_samples={num_samples}"
        if features.strip():
            dataset_requirements += f", features={features}"
        agent = DatasetAgent(prompt, dataset_requirements, int(num_samples))
        dataset_X, dataset_y, local_reason = agent.run()
        local_dataset = pd.concat([dataset_X, dataset_y.rename("target")], axis=1)
    elif dataset_source == "Upload CSV":
        local_dataset, local_reason = load_csv_file(csv_file)
    elif dataset_source == "SQLite Database":
        local_dataset, local_reason = load_sqlite_db(db_file, table_name)
    
    if local_dataset is None:
        return None, None, f"Error: {local_reason}", gr.update(selected=0) # Stay on current tab if error

    # Generate metadata for the acquired dataset
    # We pass None for target_column here, as it's optional for initial metadata and will be used later.
    temp_preprocess_agent = PreprocessingAgent(local_dataset, task_description=prompt, target_column=None) 
    local_metadata = temp_preprocess_agent.generate_metadata()

    output_message = [
        f"**Dataset Acquisition Complete!**",
        f"Dataset Reason: {local_reason}",
        f"Original Dataset Shape: {local_dataset.shape}",
        f"Dataset Metadata:",
        f"- Shape: {local_metadata['shape']}",
        f"- Columns and Data Types:\n{local_metadata['dtypes']}",
        f"- Summary Statistics:\n{local_metadata['summary_stats']}",
        f"- Missing Values:\n{local_metadata['missing_values']}"
    ]
    
    return local_dataset, local_metadata, "\n".join(output_message), gr.update(selected=1) # Move to Preprocessing tab

# --- New Function for Preprocessing and EDA Step ---
def process_preprocessing_and_eda(current_dataset_state, current_metadata_state, prompt, target_column, ordinal_columns):
    """
    Performs preprocessing and EDA on the acquired dataset.
    """
    if current_dataset_state is None:
        return "Please acquire a dataset first!", None, gr.update(selected=0)

    # Use the actual target column for the PreprocessingAgent
    preprocess_agent = PreprocessingAgent(current_dataset_state, 
                                          task_description=prompt, 
                                          target_column=target_column if target_column.strip() else None)
    
    ordinal_cols = [col.strip() for col in ordinal_columns.split(",") if col.strip()] if ordinal_columns else None
    
    # Run preprocessing and EDA
    preprocessed_df, preprocess_reason_full = preprocess_agent.run(ordinal_columns=ordinal_cols)
    
    output_message = [
        f"**Preprocessing & EDA Complete!**",
        f"Preprocessed Dataset Shape: {preprocessed_df.shape}",
        f"Preprocessing and EDA Details:",
        f"{preprocess_reason_full}"
    ]
    
    # Return preprocessed dataset, metadata, and the full reason for output
    return preprocessed_df, current_metadata_state, "\n".join(output_message)


with gr.Blocks() as demo:
    gr.Markdown("# AutoML Dataset Generator, EDA, and Preprocessor")

    # Gradio State variables to hold data between function calls
    # Using None as initial value, which means no dataset is loaded yet.
    dataset_state = gr.State(value=global_dataset) 
    metadata_state = gr.State(value=global_metadata)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("1. Data Acquisition", id=0):
            gr.Markdown("## Load or Generate Your Dataset")
            dataset_source = gr.Radio(
                choices=["Generate Synthetic Dataset", "Upload CSV", "SQLite Database"],
                label="Dataset Source",
                value="Generate Synthetic Dataset"
            )
            
            with gr.Group(visible=True) as synthetic_group:
                prompt_input_acq = gr.Textbox(label="Task Description (for synthetic data)", placeholder="e.g., classification model for customer churn")
                num_samples_input = gr.Number(label="Number of Samples", value=1000, precision=0)
                features_input = gr.Textbox(label="Desired Features (optional)", placeholder="e.g., age, tenure, contract_type")
            
            with gr.Group(visible=False) as csv_group:
                csv_file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            
            with gr.Group(visible=False) as db_group:
                db_file_input = gr.File(label="Upload SQLite Database File", file_types=[".db", ".sqlite"])
                table_name_input = gr.Textbox(label="Table Name", placeholder="e.g., customers")
            
            acquire_button = gr.Button("Acquire Dataset")
            acquisition_output = gr.Textbox(label="Dataset Acquisition Status", lines=10)
            
            # Link toggle function to dataset_source radio button
            dataset_source.change(
                fn=lambda ds: {
                    synthetic_group: gr.update(visible=ds == "Generate Synthetic Dataset"),
                    csv_group: gr.update(visible=ds == "Upload CSV"),
                    db_group: gr.update(visible=ds == "SQLite Database")
                },
                inputs=dataset_source,
                outputs=[synthetic_group, csv_group, db_group]
            )
            
            # Link acquisition button to the new function
            acquire_button.click(
                fn=process_dataset_acquisition,
                inputs=[
                    dataset_source,
                    prompt_input_acq, # Using a separate prompt input for acquisition stage
                    num_samples_input,
                    features_input,
                    csv_file_input,
                    db_file_input,
                    table_name_input,
                    dataset_state, # Pass current state for potential reuse/overwrite
                    metadata_state # Pass current state for potential reuse/overwrite
                ],
                outputs=[dataset_state, metadata_state, acquisition_output, tabs] # Update state, output, and current tab
            )

        with gr.TabItem("2. Preprocessing & EDA", id=1):
            gr.Markdown("## Preprocess Your Dataset and Explore Data")
            gr.Markdown("### Dataset Details (from Acquisition Step):")
            current_dataset_info = gr.Markdown("No dataset loaded yet. Please go back to 'Data Acquisition' tab.")
            
            # Display metadata from dataset_state after acquisition
            # This will be updated by the process_dataset_acquisition function
            def update_dataset_info(dataset_state, metadata_state):
                if dataset_state is not None and metadata_state is not None:
                    info_str = [
                        f"**Current Dataset Loaded:** Yes",
                        f"**Shape:** {dataset_state.shape}",
                        f"**Metadata:**",
                        f"- Shape: {metadata_state['shape']}",
                        f"- Columns and Data Types:\n```\n{metadata_state['dtypes']}\n```", # Use code block for dtypes
                        f"- Summary Statistics:\n```\n{metadata_state['summary_stats']}\n```",
                        f"- Missing Values:\n```\n{metadata_state['missing_values']}\n```"
                    ]
                    return "\n".join(info_str)
                return "No dataset loaded yet. Please go back to 'Data Acquisition' tab."

            # Define inputs for preprocessing
            prompt_input_eda = gr.Textbox(label="Task Description (for preprocessing/EDA recommendations)", placeholder="e.g., classification model for customer churn, aiming for high accuracy with SVM.")
            target_column_input = gr.Textbox(label="Target Column (optional, but recommended for supervised tasks)", placeholder="e.g., churn")
            ordinal_columns_input = gr.Textbox(label="Ordinal Columns (comma-separated, optional)", placeholder="e.g., education_level, satisfaction_score")
            
            preprocess_button = gr.Button("Perform Preprocessing and EDA")
            preprocessing_output = gr.Textbox(label="Preprocessing & EDA Status", lines=20)

            # Update dataset info when dataset_state changes
            dataset_state.change(
                fn=update_dataset_info,
                inputs=[dataset_state, metadata_state],
                outputs=current_dataset_info
            )
            
            # Link preprocessing button to the new function
            preprocess_button.click(
                fn=process_preprocessing_and_eda,
                inputs=[
                    dataset_state,
                    metadata_state, # Pass metadata along
                    prompt_input_eda,
                    target_column_input,
                    ordinal_columns_input
                ],
                outputs=[dataset_state, metadata_state, preprocessing_output] # Update state and output
            )

    # Note: `demo.launch()` is already at the end of your original script.
demo.launch()