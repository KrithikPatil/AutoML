import gradio as gr
import pandas as pd
import sqlite3
from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.modeling_agent import ModelingAgent
import os # Already imported
import glob # For finding image files

# Define constants for EDA artifact paths
EDA_BASE_DIR = "dataset"
EDA_PLOTS_DIR = os.path.join(EDA_BASE_DIR, "eda")
EDA_TEXT_REPORT_PATH = os.path.join(EDA_BASE_DIR, "eda_results.txt")


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
        return None, None, "Error: Please acquire a dataset first in Tab 1.", "No EDA report available.", None, gr.update(selected=0)

    # Use the actual target column for the PreprocessingAgent
    preprocess_agent = PreprocessingAgent(current_dataset_state, 
                                          task_description=prompt, 
                                          target_column=target_column if target_column.strip() else None)
    
    ordinal_cols = [col.strip() for col in ordinal_columns.split(",") if col.strip()] if ordinal_columns else None
    try:
        # Run preprocessing and EDA. The agent saves EDA artifacts to files.
        # The second return value is now a list of preprocessing reasons.
        preprocessed_df, preprocess_reason_list = preprocess_agent.run(ordinal_columns=ordinal_cols)
    except Exception as e:
        error_msg = f"Error during preprocessing/EDA: {str(e)}"
        return current_dataset_state, current_metadata_state, error_msg, error_msg, None # Keep current dataset, show error

    # 1. Construct the preprocessing log text
    preprocessing_log_text = "**Preprocessing Log:**\n" + "\n".join(preprocess_reason_list)

    # 2. Read the textual EDA summary from the file
    eda_summary_text = f"EDA textual report ({EDA_TEXT_REPORT_PATH}) not found or could not be read."
    if os.path.exists(EDA_TEXT_REPORT_PATH):
        try:
            with open(EDA_TEXT_REPORT_PATH, 'r', encoding='utf-8') as f:
                eda_summary_text = f.read()
                if not eda_summary_text.strip():
                    eda_summary_text = "EDA report file is empty."
        except Exception as e:
            eda_summary_text = f"Error reading EDA report file '{EDA_TEXT_REPORT_PATH}': {str(e)}"
            
    # 3. Collect paths to all generated EDA plots
    eda_image_paths = []
    if os.path.exists(EDA_PLOTS_DIR):
        eda_image_paths = sorted(glob.glob(os.path.join(EDA_PLOTS_DIR, "*.png"))) # Sort for consistent order

    # Return preprocessed dataset, metadata, preprocessing log, EDA text, and EDA plot paths
    return preprocessed_df, current_metadata_state, preprocessing_log_text, eda_summary_text, eda_image_paths

# --- New Function for Modeling Step ---
def process_modeling(current_dataset_state, prompt, target_column, model_choice):
    """
    Trains a model on the preprocessed data and returns the evaluation report.
    """
    if current_dataset_state is None:
        return "Error: No preprocessed dataset available. Please complete Step 2 first."

    if not target_column or not target_column.strip():
        return "Error: Target column must be specified for modeling."

    try:
        # The dataset in current_dataset_state is already preprocessed
        modeling_agent = ModelingAgent(
            dataset=current_dataset_state,
            task_description=prompt,
            target_column=target_column,
            model_choice=model_choice # Pass the selected model choice
        )
        report = modeling_agent.run()
        return report
    except Exception as e:
        return f"An unexpected error occurred during modeling: {str(e)}"

# Define update_dataset_info as a top-level function
def update_dataset_info(dataset_state, metadata_state):
    """
    Updates the Markdown display for dataset information in the Preprocessing tab.
    """
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

# Define update_preprocessed_info as a top-level function
def update_preprocessed_info(dataset_state, target_column):
    """
    Updates the Markdown display for preprocessed dataset information in the Modeling tab.
    """
    if dataset_state is not None:
        target_col_info = "Target column not specified or found."
        if target_column and target_column in dataset_state.columns:
            target_col_info = f"Target column '{target_column}' found. Dtype: {dataset_state[target_column].dtype}, Unique values: {dataset_state[target_column].nunique()}"
        info_str = [
            f"**Preprocessed Dataset Loaded:** Yes",
            f"**Shape:** {dataset_state.shape}",
            f"**Columns:** {', '.join(dataset_state.columns)}",
            f"**Info:** {target_col_info}"
        ]
        return "\n".join(info_str)
    return "No preprocessed data available. Please complete Step 2."

with gr.Blocks() as demo:
    gr.Markdown("# AutoML Dataset Generator, EDA, and Preprocessor")

    # Gradio State variables to hold data between function calls
    # Using None as initial value, which means no dataset is loaded yet.
    dataset_state = gr.State(value=None) 
    metadata_state = gr.State(value=None)
    
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
            # Define inputs for preprocessing
            prompt_input_eda = gr.Textbox(label="Task Description (for preprocessing/EDA recommendations)", placeholder="e.g., classification model for customer churn, aiming for high accuracy with SVM.")
            target_column_input = gr.Textbox(label="Target Column (optional, but recommended for supervised tasks)", placeholder="e.g., churn")
            ordinal_columns_input = gr.Textbox(label="Ordinal Columns (comma-separated, optional)", placeholder="e.g., education_level, satisfaction_score")
            
            preprocess_button = gr.Button("Perform Preprocessing and EDA")
            
            gr.Markdown("### Preprocessing Log")
            preprocessing_log_output = gr.Textbox(label="Steps taken by the Preprocessing Agent", lines=10, interactive=False, show_copy_button=True)

            gr.Markdown("### EDA Textual Summary")
            eda_text_output = gr.Textbox(label="Exploratory Data Analysis Report", lines=15, interactive=False, show_copy_button=True)

            gr.Markdown("### EDA Visualizations")
            eda_plots_output = gr.Gallery(
                label="Generated EDA Plots", show_label=False, elem_id="eda_gallery",
                columns=[2], height="auto", object_fit="contain"
            )

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
                outputs=[
                    dataset_state, 
                    metadata_state, 
                    preprocessing_log_output, 
                    eda_text_output, 
                    eda_plots_output
                ] # Update state and all relevant outputs
            )

        with gr.TabItem("3. Modeling", id=2):
            gr.Markdown("## Train a Model")
            gr.Markdown("This step uses the preprocessed data from the previous tab to train a model (RandomForest) and evaluate its performance.")
            
            gr.Markdown("### Preprocessed Dataset Info:")
            preprocessed_dataset_info = gr.Markdown("No preprocessed data available. Please complete Step 2.")

            # This function will update the info when the tab is selected or data changes
            # This function needs to be defined within the scope where dataset_state and target_column_input are available,
            # or passed as arguments. It's currently fine as it is.
            
            # Add a dropdown for model selection
            model_choice_input = gr.Dropdown(
                choices=["Random Forest", "Logistic Regression", "Linear Regression", "Gradient Boosting"],
                label="Select Model",
                value="Random Forest", # Default selection
                info="Choose a machine learning model for training."
            )

            # Update the info display when dataset_state changes
            dataset_state.change(
                fn=update_preprocessed_info,
                inputs=[dataset_state, target_column_input],
                outputs=preprocessed_dataset_info
            )

            train_button = gr.Button("Train Model and Evaluate")
            
            modeling_results_output = gr.Textbox(
                label="Modeling Results",
                lines=20,
                interactive=False,
                show_copy_button=True
            )

            train_button.click(
                fn=process_modeling,
                # Inputs should come from the EDA tab, as they define the task
                inputs=[
                    dataset_state,
                    prompt_input_eda,
                    target_column_input,
                    model_choice_input # Add model choice to inputs
                ],
                outputs=modeling_results_output
            )

    # Note: `demo.launch()` is already at the end of your original script.
if __name__ == "__main__":
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True) # Ensure EDA plots directory exists
demo.launch()