import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ModelingAgent:
    def __init__(self, dataset, task_description, target_column, model_choice="Random Forest"):
        self.dataset = dataset.copy()
        self.task_description = task_description
        self.target_column = target_column
        self.model = None
        self.report = []
        self.model_choice = model_choice

    def _infer_ml_task(self):
        """Infer ML task (classification or regression) from task_description."""
        task_desc_lower = self.task_description.lower()
        if "classification" in task_desc_lower or "predict categories" in task_desc_lower or "churn" in task_desc_lower or "fraud" in task_desc_lower or "spam" in task_desc_lower:
            return "classification"
        elif "regression" in task_desc_lower or "predict value" in task_desc_lower or "sales" in task_desc_lower or "price" in task_desc_lower or "demand" in task_desc_lower:
            return "regression"
        
        # Fallback: Infer from target column properties if not explicit in description
        if self.target_column and self.target_column in self.dataset.columns:
            target_series = self.dataset[self.target_column]
            if pd.api.types.is_numeric_dtype(target_series):
                # Heuristic: If a numeric target has few unique values, it's likely a classification task (e.g., 0, 1, 2)
                if target_series.nunique() < 20 and target_series.dtype in ['int64', 'int32']:
                    self.report.append("Inferred classification from numeric target with few unique values.")
                    return "classification"
                self.report.append("Inferred regression from numeric target.")
                return "regression"
            else:
                self.report.append("Inferred classification from non-numeric target.")
                return "classification"
        return "unknown"

    def run(self):
        """Execute the model training and evaluation pipeline."""
        self.report.append("--- Modeling Agent Report ---")

        # 1. Validate inputs
        if self.dataset is None or self.dataset.empty:
            return "Error: Dataset is empty. Cannot proceed with modeling."
        if not self.target_column or self.target_column not in self.dataset.columns:
            return f"Error: Target column '{self.target_column}' not found in the dataset."
        
        self.report.append(f"Dataset shape: {self.dataset.shape}")
        self.report.append(f"Target column: '{self.target_column}'")

        # 2. Infer ML task
        task = self._infer_ml_task()
        self.report.append(f"Inferred ML Task: {task}")

        if task == "unknown":
            return "Error: Could not determine ML task (classification/regression) from the description or target column."

        # 3. Prepare data
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if not non_numeric_cols.empty:
            return f"Error: Non-numeric feature columns found after preprocessing: {list(non_numeric_cols)}. Please check preprocessing steps."

        # 4. Split data
        try:
            if task == "classification":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.report.append(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")
        except ValueError as e:
             return f"Error during data split. This can happen if a class has too few members for stratification. Error: {e}"

        # 5. Select, train, and evaluate model
        try:
            if task == "classification":
                if self.model_choice == "Random Forest":
                    self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
                elif self.model_choice == "Logistic Regression":
                    self.model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
                elif self.model_choice == "Gradient Boosting":
                    self.model = GradientBoostingClassifier(random_state=42)
                else:
                    return f"Error: Unknown classification model choice '{self.model_choice}'."
                
                self.report.append(f"Selected model: {self.model_choice} Classifier")
                self.model.fit(X_train, y_train)
                self.report.append("Model training complete.")
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                
                self.report.append("\n--- Evaluation Results ---")
                self.report.append(f"Accuracy: {accuracy:.4f}")
                self.report.append("\nClassification Report:\n" + class_report)

            elif task == "regression":
                if self.model_choice == "Random Forest":
                    self.model = RandomForestRegressor(random_state=42, n_jobs=-1)
                elif self.model_choice == "Linear Regression":
                    self.model = LinearRegression()
                elif self.model_choice == "Gradient Boosting":
                    self.model = GradientBoostingRegressor(random_state=42)
                else:
                    return f"Error: Unknown regression model choice '{self.model_choice}'."

                self.report.append(f"Selected model: {self.model_choice} Regressor")
                self.model.fit(X_train, y_train)
                self.report.append("Model training complete.")
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                self.report.append("\n--- Evaluation Results ---")
                self.report.append(f"Mean Squared Error (MSE): {mse:.4f}")
                self.report.append(f"R-squared (R²): {r2:.4f}")
            else:
                return "Error: ML task could not be determined for modeling."
        except Exception as e:
            return f"An error occurred during model training or evaluation: {str(e)}"
        
        return "\n".join(self.report)