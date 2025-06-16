import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from dotenv import load_dotenv
import os
import ollama
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import re # Import regex for task inference

# Load environment variables
load_dotenv()

PREPROCESSING_PROMPT_TEMPLATE = """
You are an expert in machine learning data preprocessing.

Based on the task description and dataset metadata below, recommend specific data preprocessing techniques required for the requested model. Provide a concise list of techniques (e.g., handle missing values, encode categorical variables, scale numerical features, remove outliers) tailored to the model and dataset characteristics. Do not include train-test split as a preprocessing step.

Task Description: {task_description}

Dataset Metadata:
- Shape: {shape}
- Columns and Data Types: {dtypes}
- Summary Statistics: {summary_stats}
- Missing Values: {missing_values}

Output a list of preprocessing techniques in the following format:
- Technique 1
- Technique 2
- ...

Example:
- Handle missing values with mean imputation for numerical and mode for categorical
- One-hot encode categorical variables
- Scale numerical features using StandardScaler
- Remove outliers using z-score method
"""

class PreprocessingAgent:
    def __init__(self, dataset, task_description, target_column=None):
        self.dataset = dataset.copy()
        self.task_description = task_description
        self.target_column = target_column
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoders = {}
        self.numerical_columns = []
        self.categorical_columns = []
        self.reason = []
        self.eda_results = []
        self.inferred_task = self._infer_ml_task() # New: Infer ML task
        self._coerce_types() # NEW: Coerce types aggressively at initialization
        self.identify_columns() # Call identify_columns after type coercion

    def _coerce_types(self):
        """Attempt to coerce columns to numeric types where possible."""
        initial_dtypes = self.dataset.dtypes.to_dict()
        for col in self.dataset.columns:
            # Skip the target column if it's already identified and not meant to be numeric
            # (e.g., if it's a categorical target for classification)
            if self.target_column and col == self.target_column and self.inferred_task == "classification":
                continue 
            
            # Attempt to convert to numeric, coercing errors will turn invalid parsing into NaN
            converted_series = pd.to_numeric(self.dataset[col], errors='coerce')
            
            # Check if conversion actually changed the dtype to numeric and didn't introduce too many NaNs
            # If the original was not numeric and the new is numeric, and it's mostly numeric
            if pd.api.types.is_object_dtype(initial_dtypes.get(col)) and pd.api.types.is_numeric_dtype(converted_series.dtype) and converted_series.notna().sum() / len(converted_series) > 0.8: # Require >80% non-NaN after conversion
                self.dataset[col] = converted_series
                self.reason.append(f"Coerced column '{col}' to numeric type.")
            # Special handling for boolean-like columns that might become objects if mixed with other types
            elif self.dataset[col].nunique(dropna=False) <= 2 and not pd.api.types.is_numeric_dtype(self.dataset[col]):
                try:
                    # Attempt to convert to boolean, then to int if needed for numerical treatment
                    self.dataset[col] = self.dataset[col].astype(bool).astype(int)
                    self.reason.append(f"Coerced binary column '{col}' to numeric (int) type.")
                except:
                    pass # Keep as object if coercion fails or is not appropriate
        self.reason.append("Completed type coercion for dataset columns.")


    def _infer_ml_task(self):
        """Infer ML task (classification, regression, clustering) from task_description."""
        task_desc_lower = self.task_description.lower()
        if "classification" in task_desc_lower or "predict categories" in task_desc_lower or "churn" in task_desc_lower or "fraud" in task_desc_lower or "spam" in task_desc_lower:
            return "classification"
        elif "regression" in task_desc_lower or "predict value" in task_desc_lower or "sales" in task_desc_lower or "price" in task_desc_lower or "demand" in task_desc_lower:
            return "regression"
        elif "cluster" in task_desc_lower or "segment" in task_desc_lower or "group" in task_desc_lower:
            return "clustering"
        return "unknown"

    def identify_columns(self):
        """Identify numerical and categorical columns in the dataset."""
        self.numerical_columns = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if it exists and is not numerical/categorical (e.g., if it's mixed type or ID)
        if self.target_column:
            if self.target_column in self.numerical_columns:
                self.numerical_columns.remove(self.target_column)
            elif self.target_column in self.categorical_columns:
                self.categorical_columns.remove(self.target_column)
            else:
                # If target is not in identified numerical/categorical, check if it's in all columns
                if self.target_column not in self.dataset.columns:
                    self.reason.append(f"Warning: Target column '{self.target_column}' not found in dataset. Skipping target-specific analysis.")
                    self.target_column = None # Clear target if not found
        
        self.reason.append(f"Identified {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical columns.")

    def generate_metadata(self):
        """Generate metadata about the dataset."""
        metadata = {}
        metadata['shape'] = f"{self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns"
        
        # Data types
        dtypes = self.dataset.dtypes.to_dict()
        dtypes_str = "\n".join([f"{col}: {dtype}" for col, dtype in dtypes.items()])
        metadata['dtypes'] = dtypes_str
        
        # Summary statistics
        summary_stats = []
        for col in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[col]): # Use direct check for current type
                stats = self.dataset[col].describe()
                stats_str = f"Column '{col}': mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
                summary_stats.append(stats_str)
            elif pd.api.types.is_string_dtype(self.dataset[col]) or pd.api.types.is_categorical_dtype(self.dataset[col]):
                unique_vals = self.dataset[col].nunique()
                top_vals = self.dataset[col].value_counts().head(3).to_dict()
                top_vals_str = ", ".join([f"{val}: {count}" for val, count in top_vals.items()])
                stats_str = f"Column '{col}': {unique_vals} unique values, top values: {top_vals_str}"
                summary_stats.append(stats_str)
        metadata['summary_stats'] = "\n".join(summary_stats) if summary_stats else "No summary statistics available."
        
        # Missing values
        missing_counts = self.dataset.isnull().sum()
        missing_summary = missing_counts[missing_counts > 0].to_dict()
        missing_str = "\n".join([f"{col}: {count}" for col, count in missing_summary.items()]) if missing_summary else "No missing values."
        metadata['missing_values'] = missing_str
        
        return metadata

    def perform_eda(self):
        """Perform Exploratory Data Analysis and save results with dynamic plotting."""
        self.eda_results.append("Exploratory Data Analysis Results:")
        
        # Summary statistics
        summary_stats = self.dataset.describe(include='all').to_string()
        self.eda_results.append("\nSummary Statistics:\n" + summary_stats)
        
        # Data types
        dtypes = self.dataset.dtypes.to_string()
        self.eda_results.append("\nData Types:\n" + dtypes)
        
        # Missing values
        missing_counts = self.dataset.isnull().sum()
        missing_summary = missing_counts[missing_counts > 0].to_string() if missing_counts.sum() > 0 else "No missing values."
        self.eda_results.append("\nMissing Values:\n" + missing_summary)
        
        # Outliers detection using z-score for numerical columns
        outliers_summary = []
        for col in self.numerical_columns:
            # Only check for outliers if column is not empty after dropping NaNs
            if not self.dataset[col].dropna().empty: 
                z_scores = np.abs(zscore(self.dataset[col].dropna()))
                outlier_count = (z_scores > 3).sum()
                if outlier_count > 0:
                    outliers_summary.append(f"Column '{col}' has {outlier_count} outliers (z-score > 3).")
        if outliers_summary:
            self.eda_results.append("\nOutliers:\n" + "\n".join(outliers_summary))
        else:
            self.eda_results.append("\nOutliers: None detected.")
        
        # Visualizations
        eda_dir = os.path.join("dataset", "eda")
        os.makedirs(eda_dir, exist_ok=True)
        
        # Correlation heatmap for numerical columns
        if len(self.numerical_columns) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.dataset[self.numerical_columns].corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            corr_heatmap_path = os.path.join(eda_dir, "correlation_heatmap.png")
            plt.savefig(corr_heatmap_path)
            plt.close()
            self.eda_results.append(f"Saved correlation heatmap to {corr_heatmap_path}")
        
        # Histograms for numerical columns
        for col in self.numerical_columns:
            if not self.dataset[col].dropna().empty:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.dataset[col], kde=True)
                plt.title(f"Distribution of {col}")
                hist_path = os.path.join(eda_dir, f"histogram_{col}.png")
                plt.savefig(hist_path)
                plt.close()
                self.eda_results.append(f"Saved histogram for '{col}' to {hist_path}")

        # Bar plots for categorical columns
        for col in self.categorical_columns:
            if not self.dataset[col].dropna().empty:
                plt.figure(figsize=(10, 6))
                sns.countplot(y=self.dataset[col], order=self.dataset[col].value_counts().index)
                plt.title(f"Value Counts of {col}")
                plt.tight_layout()
                bar_path = os.path.join(eda_dir, f"bar_plot_{col}.png")
                plt.savefig(bar_path)
                plt.close()
                self.eda_results.append(f"Saved bar plot for '{col}' to {bar_path}")

        # Dynamic plotting based on inferred task and target column
        if self.target_column and self.target_column in self.dataset.columns:
            target_dtype = self.dataset[self.target_column].dtype
            
            # If target is numerical (likely regression)
            if pd.api.types.is_numeric_dtype(target_dtype):
                self.eda_results.append(f"\nTarget Column '{self.target_column}' (Regression Task):")
                
                # Distribution of target variable
                plt.figure(figsize=(8, 6))
                sns.histplot(self.dataset[self.target_column], kde=True)
                plt.title(f"Distribution of Target: {self.target_column}")
                target_hist_path = os.path.join(eda_dir, f"target_distribution_{self.target_column}.png")
                plt.savefig(target_hist_path)
                plt.close()
                self.eda_results.append(f"Saved target distribution histogram to {target_hist_path}")

                # Scatter plots of numerical features vs. target
                for col in self.numerical_columns:
                    if col != self.target_column and not self.dataset[[col, self.target_column]].dropna().empty:
                        plt.figure(figsize=(8, 6))
                        sns.scatterplot(x=self.dataset[col], y=self.dataset[self.target_column])
                        plt.title(f"{col} vs. {self.target_column}")
                        scatter_path = os.path.join(eda_dir, f"scatter_{col}_vs_target.png")
                        plt.savefig(scatter_path)
                        plt.close()
                        self.eda_results.append(f"Saved scatter plot for '{col}' vs. target to {scatter_path}")

            # If target is categorical (likely classification)
            elif pd.api.types.is_string_dtype(target_dtype) or pd.api.types.is_categorical_dtype(target_dtype):
                self.eda_results.append(f"\nTarget Column '{self.target_column}' (Classification Task):")
                
                # Bar plot of target variable distribution
                plt.figure(figsize=(8, 6))
                sns.countplot(y=self.dataset[self.target_column], order=self.dataset[self.target_column].value_counts().index)
                plt.title(f"Distribution of Target Classes: {self.target_column}")
                target_bar_path = os.path.join(eda_dir, f"target_class_distribution_{self.target_column}.png")
                plt.savefig(target_bar_path)
                plt.close()
                self.eda_results.append(f"Saved target class distribution bar plot to {target_bar_path}")

                # Box plots or violin plots for numerical features by target classes
                for col in self.numerical_columns:
                    if col != self.target_column and not self.dataset[[col, self.target_column]].dropna().empty:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=self.dataset[self.target_column], y=self.dataset[col])
                        plt.title(f"Distribution of {col} by {self.target_column} Class")
                        plt.tight_layout()
                        box_plot_path = os.path.join(eda_dir, f"boxplot_{col}_by_target.png")
                        plt.savefig(box_plot_path)
                        plt.close()
                        self.eda_results.append(f"Saved box plot for '{col}' by target to {box_plot_path}")
                
                # Stacked bar plots for categorical features by target classes
                for col in self.categorical_columns:
                    if col != self.target_column and not self.dataset[[col, self.target_column]].dropna().empty:
                        plt.figure(figsize=(10, 7))
                        # Use pd.crosstab for counts and then plot
                        cross_tab = pd.crosstab(self.dataset[col], self.dataset[self.target_column])
                        cross_tab.plot(kind='bar', stacked=True, figsize=(10, 7))
                        plt.title(f"Distribution of {col} across {self.target_column} Classes")
                        plt.xlabel(col)
                        plt.ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        stacked_bar_path = os.path.join(eda_dir, f"stacked_bar_{col}_by_target.png")
                        plt.savefig(stacked_bar_path)
                        plt.close()
                        self.eda_results.append(f"Saved stacked bar plot for '{col}' by target to {stacked_bar_path}")

        # Save EDA results to file
        eda_text_path = os.path.join("dataset", "eda_results.txt")
        with open(eda_text_path, 'w', encoding='utf-8') as f: # Added encoding
            f.write("\n".join(self.eda_results))
        self.reason.append(f"Saved EDA results to {eda_text_path}.")
        self.reason.append(f"EDA plots saved in '{eda_dir}' directory.")


    def query_llm_for_preprocessing(self):
        """Query LLM for preprocessing techniques based on task description and metadata."""
        try:
            metadata = self.generate_metadata() # Re-generate metadata to ensure it's up-to-date
            prompt = PREPROCESSING_PROMPT_TEMPLATE.format(
                task_description=self.task_description,
                shape=metadata['shape'],
                dtypes=metadata['dtypes'],
                summary_stats=metadata['summary_stats'],
                missing_values=metadata['missing_values']
            )
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"num_predict": 200}
            )
            techniques = [line.strip("- ").strip() for line in response["response"].strip().split("\\n") if line.startswith("- ")]
            self.reason.append(f"LLM recommended preprocessing techniques: {', '.join(techniques)}")
            return techniques
        except Exception as e:
            self.reason.append(f"LLM query failed: {str(e)}. Using default preprocessing.")
            # Default preprocessing based on common ML needs
            default_techniques = []
            if self.dataset.isnull().any().any(): # Check if there are any missing values at all
                default_techniques.append("Handle missing values with mean imputation for numerical and mode for categorical")
            if len(self.categorical_columns) > 0:
                default_techniques.append("One-hot encode categorical variables")
            if len(self.numerical_columns) > 0:
                default_techniques.append("Scale numerical features using StandardScaler")
            return default_techniques if default_techniques else ["No specific preprocessing recommended (dataset appears clean)."]


    def handle_missing_values(self):
        """Impute missing values: mean for numerical, mode for categorical."""
        for col in self.numerical_columns:
            if self.dataset[col].isnull().any():
                mean_value = self.dataset[col].mean()
                self.dataset[col].fillna(mean_value, inplace=True)
                self.reason.append(f"Imputed missing values in numerical column '{col}' with mean {mean_value:.2f}.")
        
        for col in self.categorical_columns:
            if self.dataset[col].isnull().any():
                # Ensure the column is of 'category' dtype before getting mode for robustness
                # self.dataset[col] = self.dataset[col].astype('category') 
                mode_value = self.dataset[col].mode()[0]
                self.dataset[col].fillna(mode_value, inplace=True)
                self.reason.append(f"Imputed missing values in categorical column '{col}' with mode '{mode_value}'.")

    def encode_categorical(self, ordinal_columns=None):
        """Encode categorical variables: one-hot for nominal, label encoding for ordinal."""
        ordinal_columns = ordinal_columns or []
        
        nominal_columns = [col for col in self.categorical_columns if col not in ordinal_columns]
        
        # Ensure target column is not accidentally one-hot encoded if it's categorical but not specified as ordinal
        if self.target_column and self.target_column in nominal_columns:
            nominal_columns.remove(self.target_column)

        if nominal_columns:
            # Check if there are valid nominal columns to encode
            if not self.dataset[nominal_columns].empty:
                # Need to reset index to avoid alignment issues after drop
                self.dataset.reset_index(drop=True, inplace=True) 
                
                # Fit and transform only on non-missing nominal data
                encoded_data = self.onehot_encoder.fit_transform(self.dataset[nominal_columns].astype(str)) # Convert to string to avoid DTypeWarning
                encoded_columns = self.onehot_encoder.get_feature_names_out(nominal_columns)
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=self.dataset.index)
                
                self.dataset = pd.concat([self.dataset.drop(columns=nominal_columns), encoded_df], axis=1)
                self.reason.append(f"One-hot encoded nominal columns: {nominal_columns}.")
            else:
                self.reason.append(f"No valid nominal columns to one-hot encode, or they are empty: {nominal_columns}")
        
        for col in ordinal_columns:
            if col in self.categorical_columns and col != self.target_column: # Exclude target column from label encoding if it's there
                # Ensure the column is not empty after dropping NaNs
                if not self.dataset[col].dropna().empty:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit and transform only on non-missing ordinal data
                    self.dataset[col] = self.label_encoders[col].fit_transform(self.dataset[col].astype(str)) # Convert to string to avoid DTypeWarning
                    self.reason.append(f"Label encoded ordinal column: {col}.")
                else:
                    self.reason.append(f"Ordinal column '{col}' is empty or contains only missing values, skipping Label Encoding.")

        # Re-identify numerical and categorical columns after encoding to ensure correctness
        self.identify_columns()


    def scale_numerical(self):
        """Scale numerical features using StandardScaler."""
        # Ensure numerical_columns are correctly identified and have data
        current_numerical_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column if it's numerical and should not be scaled (e.g., target labels)
        if self.target_column and self.target_column in current_numerical_cols:
             if self.inferred_task == "classification" and len(self.dataset[self.target_column].unique()) < 50: # Assume classification if few unique values
                 current_numerical_cols.remove(self.target_column)
             elif self.inferred_task == "regression": # For regression, target is numerical but usually not scaled with features
                 current_numerical_cols.remove(self.target_column)

        if current_numerical_cols:
            # Filter out columns that might have constant values (std=0), which break StandardScaler
            cols_to_scale = [col for col in current_numerical_cols if self.dataset[col].std() > 1e-9]
            if cols_to_scale:
                self.dataset[cols_to_scale] = self.scaler.fit_transform(self.dataset[cols_to_scale])
                self.reason.append(f"Scaled numerical columns: {cols_to_scale}.")
            else:
                self.reason.append("No numerical columns with variance to scale.")
        else:
            self.reason.append("No numerical columns found to scale.")


    def remove_outliers(self):
        """Remove outliers using z-score method for numerical columns."""
        initial_len = len(self.dataset)
        # Create a boolean mask for rows to keep
        keep_mask = pd.Series(True, index=self.dataset.index)

        for col in self.numerical_columns:
            if not self.dataset[col].dropna().empty and self.dataset[col].std() > 1e-9: # Only if data exists and is not constant
                z_scores = np.abs(zscore(self.dataset[col].dropna()))
                # Apply z-score filter only to the non-NaN values; align by index
                outlier_indices = self.dataset[col].dropna().index[z_scores > 3]
                keep_mask.loc[outlier_indices] = False # Mark these rows for removal

        self.dataset = self.dataset[keep_mask]
        removed_count = initial_len - len(self.dataset)
        if removed_count > 0:
            self.reason.append(f"Removed {removed_count} rows containing outliers from numerical columns using z-score method.")
        else:
            self.reason.append("No outliers detected or removed based on z-score > 3.")


    def run(self, ordinal_columns=None):
        """Execute EDA and preprocessing steps."""
        self.reason = [] # Reset reasons for each run
        self.eda_results = [] # Reset EDA results for each run

        # identify_columns is now called in __init__ after _coerce_types
        # self.identify_columns() 
        self.perform_eda() # Perform EDA first

        # Query LLM for preprocessing techniques with metadata
        techniques = self.query_llm_for_preprocessing()
        
        # Apply preprocessing techniques based on LLM recommendations
        for technique in techniques:
            if "missing values" in technique.lower():
                self.handle_missing_values()
            # Check for both "encode categorical" and "one-hot" as LLM might output either
            if "encode categorical" in technique.lower() or "one-hot" in technique.lower() or "label encode" in technique.lower():
                self.encode_categorical(ordinal_columns)
            # Check for "scale numerical" or specific scalers like "standardscaler"
            if "scale numerical" in technique.lower() or "standardscaler" in technique.lower():
                self.scale_numerical()
            if "remove outliers" in technique.lower() or "z-score" in technique.lower():
                self.remove_outliers()
        
        # Save preprocessed dataset
        os.makedirs("dataset", exist_ok=True)
        preprocessed_path = os.path.join("dataset", "preprocessed_dataset.csv")
        self.dataset.to_csv(preprocessed_path, index=False)
        self.reason.append(f"Saved preprocessed dataset to {preprocessed_path}.")
        
        return self.dataset, self.reason # Return the list of reasons directly