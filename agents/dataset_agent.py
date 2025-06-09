import os
import re
import pandas as pd
from sklearn.datasets import load_iris
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import kaggle

# Load environment variables
load_dotenv()
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Prompt Template
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["task_description"],
    template="""
You are a dataset recommendation agent. Based on the following machine learning task, suggest a Kaggle dataset.

Return your response in exactly this format:
Dataset: <kaggle-dataset-identifier>
Reason: <short explanation>

Task: {task_description}
"""
)

class DatasetAgent:
    def __init__(self, task_description):
        self.task_description = task_description

        # Load Gemma model locally
        model_id = "google/gemma-7b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def find_dataset(self):
        try:
            chain = PROMPT_TEMPLATE | self.llm
            output = chain.invoke({"task_description": self.task_description})
            print("Raw LLM output:\n", output)

            match = re.search(r'Dataset:\s*(.+?)\s*Reason:\s*(.+)', output, re.DOTALL)
            if match:
                dataset_name = match.group(1).strip()
                reason = match.group(2).strip()
                return dataset_name, reason
            else:
                raise ValueError("Output could not be parsed properly.")
        except Exception as e:
            print(f"Error: {e}. Falling back to default Iris dataset.")
            return None, f"Default Iris dataset used due to error: {str(e)}"

    def load_kaggle_dataset(self, dataset_name):
        try:
            kaggle.api.dataset_download_files(dataset_name, path="dataset", unzip=True)
            for file in os.listdir("dataset"):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join("dataset", file))
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                    print(f"Loaded dataset: {dataset_name}")
                    return X, y
        except Exception as e:
            print(f"Failed to download Kaggle dataset: {e}")
            return self.load_default_dataset()

    def load_default_dataset(self):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        print("Loaded default Iris dataset")
        return X, y

    def run(self):
        dataset_name, reason = self.find_dataset()
        if dataset_name:
            X, y = self.load_kaggle_dataset(dataset_name)
        else:
            X, y = self.load_default_dataset()
        return X, y, reason
