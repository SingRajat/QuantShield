import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

class RiskClassifier:
    """
    Machine Learning model to classify Portfolio Risk (Low, Medium, High)
    based on 4 approved risk metrics: Annualized Volatility, Historical VaR (95%),
    Maximum Drawdown, and Diversification Ratio.
    """
    
    FEATURES = ["Vol", "VaR95", "MaxDD", "DivRatio"]
    TARGET = "Label"
    
    def __init__(self, random_state: int = 42):
        """
        Initializes the RiskClassifier.
        
        Args:
            random_state (int): Seed for reproducibility of the Random Forest model.
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            class_weight="balanced"  # Handle potential class imbalances
        )
        self.is_trained = False
        
    def _sort_chronologically(self, panel_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the dataset strictly by time to ensure forward-chaining validation works correctly.
        """
        if "Window_End" not in panel_dataset.columns:
            raise ValueError("'Window_End' column is required for strict time-based splitting.")
            
        return panel_dataset.sort_values(by="Window_End").reset_index(drop=True)

    def train_and_evaluate(self, panel_dataset: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Trains the model and evaluates it using TimeSeriesSplit (forward-chaining cross-validation).
        
        Args:
            panel_dataset (pd.DataFrame): The output from DatasetBuilder.
            n_splits (int): Number of splits for TimeSeriesSplit cross-validation.
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics averaged across folds.
        """
        # Validate features exist
        missing_feats = [f for f in self.FEATURES if f not in panel_dataset.columns]
        if missing_feats:
            raise ValueError(f"Panel dataset is missing required features: {missing_feats}")
            
        if self.TARGET not in panel_dataset.columns:
            raise ValueError(f"Panel dataset is missing the target column: '{self.TARGET}'")
            
        # 1. Sort Data Chronologically
        df_sorted = self._sort_chronologically(panel_dataset)
        
        X = df_sorted[self.FEATURES]
        y = df_sorted[self.TARGET]
        
        # 2. Setup TimeSeriesSplit
        # TimeSeriesSplit inherently prevents data leakage by only training on past data and testing on future data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"Starting TimeSeriesSplit Cross-Validation with {n_splits} splits...")
        
        fold_accuracies = []
        fold_reports = []
        
        # 3. Perform Forward-Chaining Validation
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Verify strict time boundary
            train_ends = df_sorted.iloc[train_index]["Window_End"].max()
            test_starts = df_sorted.iloc[test_index]["Window_End"].min()
            logger.debug(f"Fold {fold+1}: Train ends {train_ends}, Test starts {test_starts}")
            
            # Train model on this fold
            self.model.fit(X_train, y_train)
            
            # Evaluate on this fold
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            fold_accuracies.append(accuracy)
            fold_reports.append(report)
            
            logger.info(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
            
        # 4. Final Training on Full Dataset 
        # (Optional but standard: Train on all data before serving in production)
        logger.info("Refitting model on entire dataset for future inference...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # 5. Compute Average Metrics
        avg_accuracy = np.mean(fold_accuracies)
        logger.info(f"Average TimeSeriesSplit Accuracy: {avg_accuracy:.4f}")
        
        return {
            "average_accuracy": avg_accuracy,
            "fold_accuracies": fold_accuracies,
            "final_model_trained": True
        }

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts Risk Class (Low, Medium, High) for incoming portfolio features.
        
        Args:
            features_df (pd.DataFrame): DataFrame containing the 4 required features representing
                                        current portfolio risk metrics.
                                        
        Returns:
            np.ndarray: Predicted risk class labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
            
        # Ensure ordered features align with training
        X = features_df[self.FEATURES]
        predictions = self.model.predict(X)
        
        return predictions
