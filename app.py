import gradio as gr
from agents.dataset_agent import DatasetAgent

# Global variables to store state
user_prompt = None
dataset_requirements = None
dataset_X = None
dataset_y = None
reason = None

def find_dataset(prompt, num_samples, features):
    global user_prompt, dataset_requirements, dataset_X, dataset_y, reason
    user_prompt = prompt
    # Format dataset requirements
    dataset_requirements = f"num_samples={num_samples}"
    if features.strip():
        dataset_requirements += f", features={features}"
    agent = DatasetAgent(user_prompt, dataset_requirements)
    dataset_X, dataset_y, reason = agent.run()
    return f"Dataset shape: {dataset_X.shape}\nReason: {reason}"

with gr.Blocks() as demo:
    gr.Markdown("# AutoML Dataset Generator")
    with gr.Row():
        prompt_input = gr.Textbox(label="Task Description", placeholder="e.g., classification model for customer churn")
        num_samples_input = gr.Number(label="Number of Samples", value=1000, precision=0)
        features_input = gr.Textbox(label="Desired Features (optional)", placeholder="e.g., age, tenure, contract_type")
    find_button = gr.Button("Generate Dataset")
    output = gr.Textbox(label="Output")
    find_button.click(
        fn=find_dataset,
        inputs=[prompt_input, num_samples_input, features_input],
        outputs=output
    )

demo.launch()