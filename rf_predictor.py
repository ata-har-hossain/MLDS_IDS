# src/rf_predictor.py - Random Forest with 100% thresholds
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_predictor import BasePredictor

class RF_Predictor(BasePredictor):
    def __init__(self, model_path='model/rf_model.pkl', anomaly_threshold=1.00):
        """
        Initialize Random Forest Predictor with 100% confidence thresholds
        
        Args:
            model_path: Path to model file
            anomaly_threshold: Fallback threshold (set to 1.00 = 100%)
        """
        super().__init__(model_path, 'Random Forest', anomaly_threshold)
        # Ensure all thresholds are 1.00
        self.normal_confidence_threshold = 1.00
        self.attack_confidence_threshold = 1.00
        self.anomaly_threshold = 1.00
        self.load_model()
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"   [WARNING] RF model not found at {self.model_path}")
                return
            
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
                print(f"   [OK] Random Forest loaded with {len(self.feature_names)} features")
            else:
                print(f"   [OK] Random Forest loaded")
            
            self.load_label_mapping()
            print(f"   [OK] STRICT MODE: All thresholds = 100%")
        except Exception as e:
            print(f"   [ERROR] RF load failed: {e}")