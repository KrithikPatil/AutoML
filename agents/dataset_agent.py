import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from dotenv import load_dotenv
import os
import ollama
from io import StringIO
import csv
import re
from scipy.stats import norm, multivariate_normal

# Load environment variables
load_dotenv()

CSV_PROMPT_TEMPLATE = """
You are a synthetic dataset generator.

Based on the task description and dataset requirements below, generate a synthetic dataset in raw CSV format with 10 sample rows and a header. You are allowed to assume and include **additional relevant features** that make sense for the given machine learning task. The dataset should be logically consistent and include realistic values.

❗ Important Instructions:
- Output only raw CSV (no markdown, no explanations, no headers like "CSV:", no quotes around the entire CSV block).
- Include a header row with clear, unique column names (e.g., use underscores instead of spaces, avoid special characters or reserved words like 'float').
- Ensure data types are appropriate (e.g., integers for IDs or counts, floats for scores, strings for text or categories).
- Enclose all string fields in double quotes to handle commas, spaces, or special characters, but do not nest quotes within quoted strings (e.g., use "Text" instead of "\"Text\"").
- Do not include missing values (NaN, empty fields, or null).
- Ensure each row has exactly the same number of fields as the header.
- Avoid leading/trailing whitespace in fields or lines.
- Specify data types in the requirements for clarity (e.g., int, float, string).

User Prompt: {task_description}
Dataset Requirements: {dataset_requirements}

Output:
"""

def clean_csv(csv_text):
    lines = csv_text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"'([^']*)'", r'"\1"', line)
        line = line.strip()
        line = line.replace("NaN", "")
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def safe_parse_csv(csv_text):
    try:
        df = pd.read_csv(
            StringIO(csv_text),
            quotechar='"',
            escapechar='\\',
            quoting=csv.QUOTE_MINIMAL,
            engine='python',
            keep_default_na=False,
            dtype=str
        )
        df = df[df.apply(lambda row: any(cell.strip() != "" for cell in row), axis=1)]
        if df.empty:
            raise ValueError("Parsed DataFrame is empty after filtering.")
        return df
    except Exception as e:
        print(f"CSV parsing failed: {e}")
        return None

def gaussian_copula_synthesize(data, n_samples, seed=42):
    np.random.seed(seed)
    data = data.copy().dropna()
    if data.empty:
        raise ValueError("Input data is empty after removing NaN values")

    marginals = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            mean, std = data[col].astype(float).mean(), data[col].astype(float).std()
            if std < 1e-10:
                std = 1e-10
            min_val, max_val = data[col].astype(float).min(), data[col].astype(float).max()
            marginals[col] = ('normal', mean, std, min_val, max_val)
        else:
            values = data[col].dropna()
            probs = values.value_counts(normalize=True).to_dict()
            marginals[col] = ('categorical', probs)

    gaussian_data = pd.DataFrame(index=data.index, columns=data.columns)
    for col, params in marginals.items():
        if params[0] == 'normal':
            mean, std, _, _ = params[1:]
            values = data[col].astype(float).clip(mean - 5*std, mean + 5*std)
            gaussian_data[col] = norm.cdf((values - mean) / std)
            gaussian_data[col] = np.clip(gaussian_data[col], 1e-6, 1-1e-6)
        else:
            probs = params[1]
            cats = sorted(probs.keys())
            cum_probs = np.cumsum([probs[cat] for cat in cats])
            gaussian_data[col] = data[col].map(
                lambda x: norm.ppf(np.clip(np.searchsorted(cum_probs, np.random.uniform()) / len(cats), 1e-6, 1-1e-6))
            )

    gaussian_data = gaussian_data.astype(float)
    cov = gaussian_data.cov() + 1e-4 * np.eye(gaussian_data.shape[1])
    mean = np.zeros(gaussian_data.shape[1])
    latent_samples = multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples, random_state=seed)
    latent_samples = pd.DataFrame(latent_samples, columns=data.columns)

    synthetic_data = latent_samples.copy()
    for col, params in marginals.items():
        if params[0] == 'normal':
            mean, std, min_val, max_val = params[1:]
            synthetic_data[col] = norm.ppf(np.clip(latent_samples[col], 1e-6, 1-1e-6)) * std + mean
            synthetic_data[col] = synthetic_data[col].clip(min_val, max_val).fillna(mean)
        else:
            probs = params[1]
            cats = sorted(probs.keys())
            cum_probs = np.cumsum([probs[cat] for cat in cats])
            synthetic_data[col] = [
                cats[np.searchsorted(cum_probs, norm.cdf(x))] for x in latent_samples[col]
            ]
    return synthetic_data

class DatasetAgent:
    def __init__(self, task_description, dataset_requirements, num_samples):
        self.task_description = task_description
        self.dataset_requirements = dataset_requirements
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.n_samples = num_samples

    def generate_csv_from_llm(self):
        try:
            prompt = CSV_PROMPT_TEMPLATE.format(
                task_description=self.task_description,
                dataset_requirements=self.dataset_requirements
            )
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"num_predict": 800}
            )
            raw_csv = response["response"].strip()
            print("Raw CSV Response:\n", raw_csv)

            cleaned_csv = clean_csv(raw_csv)
            print("Processed CSV for parsing:\n", cleaned_csv)

            df = safe_parse_csv(cleaned_csv)
            if df is None:
                raise ValueError("CSV parsing returned empty or invalid DataFrame")

            df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
            return df, "CSV-based synthetic dataset generated by LLM."
        except Exception as e:
            print(f"CSV generation failed: {e}, falling back to default Iris dataset.")
            return self.load_default_dataset(f"Default Iris dataset used due to error: {str(e)}")

    def enlarge_dataset_with_copula(self, df, n_samples):
        try:
            print(f"Generating {n_samples} rows using custom Gaussian Copula...")
            synthetic_df = gaussian_copula_synthesize(df, n_samples)
            return synthetic_df, f"Dataset enlarged to {n_samples} rows using custom Gaussian Copula."
        except Exception as e:
            print(f"Gaussian Copula enlargement failed: {e}, returning original dataset.")
            return df, f"Failed to enlarge dataset: {str(e)}"

    def load_default_dataset(self, reason="Loaded default Iris dataset."):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="target")
        df = pd.concat([X, y], axis=1)
        return df, reason

    def run(self):
        initial_df, reason = self.generate_csv_from_llm()
        enlarged_df, enlargement_reason = self.enlarge_dataset_with_copula(initial_df, self.n_samples)

        os.makedirs("dataset", exist_ok=True)
        csv_path = os.path.join("dataset", "generated_dataset.csv")
        enlarged_df.to_csv(csv_path, index=False)
        print(f"Saved dataset to {csv_path}")

        X = enlarged_df.iloc[:, :-1]
        y = enlarged_df.iloc[:, -1]
        return X, y, f"{reason} | {enlargement_reason}"
