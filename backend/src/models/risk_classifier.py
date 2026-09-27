import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

try:
    from backend.src.models.purged_cv import PurgedGroupTimeSeriesSplit
    from backend.src.models.baseline_ewma import EWMABaseline
except ImportError:
    try:
        from src.models.purged_cv import PurgedGroupTimeSeriesSplit
        from src.models.baseline_ewma import EWMABaseline
    except ImportError:
        from purged_cv import PurgedGroupTimeSeriesSplit
        from baseline_ewma import EWMABaseline

logger = logging.getLogger(__name__)

class RiskClassifier:
    """
    Machine Learning model to classify Portfolio Risk (Low, Medium, High)
    based on econometric risk metrics using CatBoost with Purged and Embargoed
    Walk-Forward Cross-Validation, rigorously benchmarked against RiskMetrics EWMA.
    """
    
    FEATURES = [
        "Vol",
        "VaR95",
        "MaxDD",
        "DivRatio",
        "Skewness",
        "Kurtosis",
        "RollingVol20",
        "RollingVol60",
        "Sharpe",
        "Sortino",
        "Beta"
    ]
    TARGET = "Label"
    
    def __init__(self, random_state: int = 42):
        """
        Initializes the RiskClassifier.
        
        Args:
            random_state (int): Seed for reproducibility.
        """
        self.model = CatBoostClassifier(
            iterations=200,
            depth=6,
            random_seed=random_state,
            verbose=0
        )
        self.is_trained = False
        self.evaluation_summary = {}
        self.ewma_baseline = EWMABaseline(decay=0.94)
        
    def _sort_chronologically(self, panel_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the dataset strictly by time to ensure forward-chaining validation works correctly.
        """
        if "Window_End" not in panel_dataset.columns:
            raise ValueError("'Window_End' column is required for strict time-based splitting.")
            
        return panel_dataset.sort_values(by="Window_End").reset_index(drop=True)

    def train_and_evaluate(self, panel_dataset: pd.DataFrame, n_splits: int = 5, embargo_days: int = 30) -> Dict[str, Any]:
        """
        Trains and evaluates the model using PurgedGroupTimeSeriesSplit.
        Benchmarked against the standard RiskMetrics EWMA baseline.
        
        Args:
            panel_dataset (pd.DataFrame): The output from DatasetBuilder.
            n_splits (int): Number of splits for cross-validation.
            embargo_days (int): Post-training buffer days to account for forward horizon and persistence.
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics and baseline comparison.
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
        
        # 2. Setup Purged & Embargoed Cross-Validation
        ptscv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
        
        logger.info(f"Starting PurgedGroupTimeSeriesSplit CV ({n_splits} splits, {embargo_days}d embargo)...")
        
        fold_accuracies = []
        fold_f1s = []
        fold_ewma_accuracies = []
        fold_reports = []
        oof_y_true = []
        oof_y_pred = []
        oof_ewma_pred = []
        
        # 3. Perform Forward-Chaining Purged Validation
        for fold, (train_index, test_index) in enumerate(ptscv.split(df_sorted)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            train_end_max = df_sorted.iloc[train_index]["Window_End"].max()
            test_start_min = df_sorted.iloc[test_index]["Window_Start"].min()
            test_end_max = df_sorted.iloc[test_index]["Window_End"].max()
            
            # Ensure fold has test samples after purging
            if len(X_test) == 0 or len(y_test) == 0:
                logger.warning(f"Fold {fold+1}: Test set is empty after purging & embargo. Skipping fold.")
                continue

            # Train model on this fold
            self.model.fit(X_train, y_train)
            
            # Evaluate ML Model on this fold
            y_pred = self.model.predict(X_test)
            if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
                y_pred = y_pred.ravel()
                
            accuracy = accuracy_score(y_test, y_pred)
            _, _, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Evaluate EWMA Baseline on this fold
            vol_proxy = X_test["RollingVol20"] if "RollingVol20" in X_test.columns else X_test["Vol"]
            ewma_preds = [self.ewma_baseline.predict_regime_from_vol(v) for v in vol_proxy]
            ewma_acc = accuracy_score(y_test, ewma_preds)
            
            fold_accuracies.append(accuracy)
            fold_f1s.append(macro_f1)
            fold_ewma_accuracies.append(ewma_acc)
            fold_reports.append(report)
            oof_y_true.extend(y_test)
            oof_y_pred.extend(y_pred)
            oof_ewma_pred.extend(ewma_preds)
            
            logger.info(
                f"Fold {fold+1} (n={len(X_test)}): ML Acc: {accuracy:.4f} | Macro F1: {macro_f1:.4f} | "
                f"EWMA Baseline Acc: {ewma_acc:.4f}"
            )
            
        # 4. Final Training on Full Dataset for serving
        logger.info("Refitting production model on entire panel dataset...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # 5. Compute Aggregate Out-of-Fold Metrics & Baseline Comparison
        avg_accuracy = float(np.mean(fold_accuracies))
        avg_macro_f1 = float(np.mean(fold_f1s))
        oof_acc = float(accuracy_score(oof_y_true, oof_y_pred))
        oof_p, oof_r, oof_f1, _ = precision_recall_fscore_support(oof_y_true, oof_y_pred, average='macro', zero_division=0)
        
        oof_ewma_acc = float(accuracy_score(oof_y_true, oof_ewma_pred))
        _, _, oof_ewma_f1, _ = precision_recall_fscore_support(oof_y_true, oof_ewma_pred, average='macro', zero_division=0)
        lift = oof_acc - oof_ewma_acc
        
        logger.info(f"Average Purged CV ML Accuracy: {avg_accuracy:.4f} | Macro F1: {avg_macro_f1:.4f}")
        logger.info(f"Out-of-Fold Aggregate ML Accuracy: {oof_acc:.4f} | OOF Macro F1: {oof_f1:.4f}")
        logger.info(f"EWMA Baseline OOF Accuracy: {oof_ewma_acc:.4f} | Baseline Macro F1: {oof_ewma_f1:.4f}")
        logger.info(f"ML Model Lift vs Statistical Baseline: {lift:+.4f} ({lift*100:+.2f}%)")
        
        self.evaluation_summary = {
            "average_accuracy": avg_accuracy,
            "average_macro_f1": avg_macro_f1,
            "oof_accuracy": oof_acc,
            "oof_macro_f1": float(oof_f1),
            "oof_macro_precision": float(oof_p),
            "oof_macro_recall": float(oof_r),
            "fold_accuracies": fold_accuracies,
            "fold_f1s": fold_f1s,
            "ewma_baseline_oof_accuracy": oof_ewma_acc,
            "ewma_baseline_oof_macro_f1": float(oof_ewma_f1),
            "model_lift_vs_baseline": float(lift),
            "final_model_trained": True
        }
        return self.evaluation_summary

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts Risk Class (Low, Medium, High) for incoming portfolio features.
        
        Args:
            features_df (pd.DataFrame): DataFrame containing required features.
                                        
        Returns:
            np.ndarray: Predicted risk class labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
            
        X = features_df[self.FEATURES]
        predictions = self.model.predict(X)
        if hasattr(predictions, 'ndim') and predictions.ndim > 1:
            predictions = predictions.ravel()
        return predictions

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Returns predicted class probabilities for incoming features.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba().")
            
        X = features_df[self.FEATURES]
        return self.model.predict_proba(X)

    def predict_risk_score(self, features_df: pd.DataFrame) -> float:
        """
        Derives a continuous 0.0-10.0 risk score from the model's predicted class probabilities.
        Provides a smooth, dynamic score without arbitrary heuristic formulas.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting risk score.")
            
        probs = self.predict_proba(features_df)[0]
        classes = list(self.model.classes_)
        
        prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
        p_low = prob_dict.get("Low", 0.0)
        p_med = prob_dict.get("Medium", 0.0)
        p_high = prob_dict.get("High", 0.0)
        
        # Smooth continuous score: Low=1.5, Med=5.0, High=8.5
        score = (p_low * 1.5) + (p_med * 5.0) + (p_high * 8.5)
        return float(np.clip(round(score, 2), 0.0, 10.0))

if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_path = project_root / 'backend' / 'src' / 'models' / 'training_dataset.csv'
    
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        classifier = RiskClassifier()
        classifier.train_and_evaluate(df, n_splits=5)
    else:
        print(f"Dataset not found at {dataset_path}. Please run train_model.py first.")
