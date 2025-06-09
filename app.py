import gradio as gr
from dotenv import load_dotenv
import os
from agents.dataset_agent import DatasetAgent

# Load environment variables from .env file
load_dotenv()

# Ensure Kaggle environment variables are set
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Store user prompt globally (for use by multiple agents)
user_prompt = None

def find_dataset(task_description):
    global user_prompt
    if not task_description:
        return "Error: Please provide a task description"
    
    # Store the user prompt
    user_prompt = task_description
    
    # Initialize and run Dataset Agent
    try:
        agent = DatasetAgent(task_description=user_prompt)
        X, y, reason = agent.run()
        
        # Prepare response
        response = (
            f"**Task Description**: {user_prompt}\n\n"
            f"**Dataset Shape**: Features: {X.shape}, Target: {y.shape}\n\n"
            f"**Reason for Selection**: {reason}"
        )
        return response
    except Exception as e:
        return f"Error: Dataset Agent failed: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AutoML Dataset Selection")
    task_input = gr.Textbox(
        label="Enter ML Task Description",
        placeholder="e.g., classification model for predicting customer churn",
        lines=4
    )
    submit_button = gr.Button("Find Dataset")
    output = gr.Markdown(label="Result")
    
    submit_button.click(
        fn=find_dataset,
        inputs=task_input,
        outputs=output
    )

if __name__ == "__main__":
    # Launch Gradio app
    demo.launch(server_name="0.0.0.0", server_port=7860)