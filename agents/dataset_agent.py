import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from dotenv import load_dotenv
import os
import re
import ollama

# Load environment variables from .env file
load_dotenv()

# Prompt for dataset feature suggestion
PROMPT_TEMPLATE = """
You are a Dataset Agent. Based on the user prompt and dataset requirements, suggest features for a synthetic dataset for a machine learning task. Return a response in Markdown format with two sections: "## Features" (a list of feature names and their data types, e.g., "age (integer), tenure (float), contract_type (categorical)") and "## Reason" (explaining the feature choices). Do not include text outside these sections.

User Prompt: {task_description}
Dataset Requirements: {dataset_requirements}

Example Response:
## Features
- age (integer)
- tenure (float)
- contract_type (categorical)
- monthly_charges (float)
- churn (binary)
## Reason
These features reflect customer attributes relevant for churn prediction, including demographic, service duration, contract, and billing information, with churn as the target.
"""

class DatasetAgent:
    def __init__(self, task_description, dataset_requirements):
        self.task_description = task_description
        self.dataset_requirements = dataset_requirements
        self.model = os.getenv("OLLAMA_MODEL", "mistral")  # Default to mistral

    def suggest_features(self):
        # Use Ollama to suggest dataset features
        try:
            prompt = PROMPT_TEMPLATE.format(
                task_description=self.task_description,
                dataset_requirements=self.dataset_requirements
            )
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"num_predict": 200}
            )
            response_text = response["response"]
            print(f"Raw LLM response: {response_text}")  # Debug output

            # Clean response
            response_text = response_text.strip()
            response_text = re.sub(r'^```markdown\s*|\s*```$', '', response_text)
            # Extract features and reason
            feature_match = re.search(r'## Features\s*([\s\S]+?)\s*## Reason\s*(.+)', response_text, re.DOTALL)
            if not feature_match:
                raise ValueError("No valid Markdown format found in response")
            features_text = feature_match.group(1).strip()
            reason = feature_match.group(2).strip()
            # Parse features into a list of (name, type)
            features = []
            for line in features_text.split('\n'):
                if line.strip().startswith('-'):
                    match = re.match(r'-\s*(\w+)\s*\((\w+)\)', line.strip())
                    if match:
                        features.append((match.group(1), match.group(2)))
            if not features:
                raise ValueError("No valid features found in response")
            print(f"Suggested features: {features}, Reason: {reason}")
            return features, reason
        except Exception as e:
            print(f"Feature suggestion failed: {e}, using default features")
            # Default features for customer churn
            default_features = [
                ("age", "integer"),
                ("tenure", "float"),
                ("contract_type", "categorical"),
                ("monthly_charges", "float"),
                ("churn", "binary")
            ]
            return default_features, f"Default features used due to error: {str(e)}"

    def create_synthetic_dataset(self, features, num_samples=1000):
        try:
            # Initialize dataset
            data = {}
            for feature_name, feature_type in features:
                if feature_type == "integer":
                    data[feature_name] = np.random.randint(18, 80, num_samples)  # e.g., age
                elif feature_type == "float":
                    if feature_name == "tenure":
                        data[feature_name] = np.random.uniform(0, 72, num_samples)
                    elif feature_name == "monthly_charges":
                        data[feature_name] = np.random.uniform(20, 120, num_samples)
                elif feature_type == "categorical":
                    data[feature_name] = np.random.choice(["Month-to-month", "One-year", "Two-year"], num_samples)
                elif feature_type == "binary":
                    data[feature_name] = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 30% churn rate
            df = pd.DataFrame(data)
            # Save to CSV
            os.makedirs('dataset', exist_ok=True)
            dataset_path = os.path.join('dataset', 'synthetic_dataset.csv')
            df.to_csv(dataset_path, index=False)
            # Split features and target (assume last feature is target)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print(f"Created synthetic dataset at {dataset_path}")
            return X, y
        except Exception as e:
            print(f"Dataset creation failed: {e}, using default Iris dataset")
            return self.load_default_dataset()

    def load_default_dataset(self):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        print("Loaded default Iris dataset")
        return X, y

    def run(self):
        # Parse dataset requirements for number of samples
        num_samples = 1000  # Default
        try:
            samples_match = re.search(r'num_samples=(\d+)', self.dataset_requirements)
            if samples_match:
                num_samples = int(samples_match.group(1))
        except Exception:
            pass
        features, reason = self.suggest_features()
        X, y = self.create_synthetic_dataset(features, num_samples)
        return X, y, reason