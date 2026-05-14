# src/gb_predictor.py - Gradient Boosting Predictor
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_predictor import BasePredictor

class GB_Predictor(BasePredictor):
    def __init__(self, model_path='model/gb_model.joblib', anomaly_threshold=1.00):
        """
        Initialize Gradient Boosting Predictor with 100% confidence thresholds
        
        Args:
            model_path: Path to model file (gb_model.joblib)
            anomaly_threshold: Fallback threshold (set to 1.00 = 100%)
        """
        super().__init__(model_path, 'Gradient Boosting', anomaly_threshold)
        # Ensure all thresholds are 1.00 (100% confidence required)
        self.normal_confidence_threshold = 1.00
        self.attack_confidence_threshold = 1.00
        self.anomaly_threshold = 1.00
        self.load_model()
    
    def load_model(self):
        """Load the Gradient Boosting model from .joblib file"""
        try:
            if not os.path.exists(self.model_path):
                print(f"   [WARNING] Gradient Boosting model not found at {self.model_path}")
                return
            
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            
            # Get feature names from model (stored similarly to Random Forest)
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
                print(f"   [OK] Gradient Boosting loaded with {len(self.feature_names)} features")
            else:
                print(f"   [OK] Gradient Boosting loaded")
            
            self.load_label_mapping()
            print(f"   [OK] STRICT MODE: All thresholds = 100% (only flags anomalies at 100% confidence)")
            
        except Exception as e:
            print(f"   [ERROR] Gradient Boosting load failed: {e}")