# src/xgb_predictor.py - XGBoost with 100% confidence requirement
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class XGB_Predictor:
    def __init__(self, model_path='model/xgb_model.json', anomaly_threshold=1.00):
        """
        Initialize XGBoost Predictor - ONLY flags anomalies at 100% confidence
        
        Args:
            model_path: Path to JSON model file (.json)
            anomaly_threshold: Threshold for anomaly detection (1.00 = 100% confidence required)
        """
        self.model_path = model_path
        self.model_name = 'XGBoost'
        self.model = None
        self.anomaly_threshold = anomaly_threshold  # Set to 1.00 for 100% confidence
        self.model_loaded = False
        self.label_mapping = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load XGBoost model from JSON format"""
        try:
            if not os.path.exists(self.model_path):
                print(f"   [ERROR] XGBoost JSON model not found at {self.model_path}")
                print(f"   [INFO] Please convert your XGBoost model to JSON format using:")
                print(f"          python -c \"import joblib; import xgboost as xgb; "
                      f"model = joblib.load('model/xgb_model.pkl'); "
                      f"model.get_booster().save_model('model/xgb_model.json')\"")
                return
            
            # Load using XGBoost's native loader (JSON format)
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self.model_loaded = True
            print(f"   [OK] XGBoost loaded from JSON: {os.path.basename(self.model_path)}")
            
            # Extract feature names from the booster
            booster = self.model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names:
                self.feature_names = booster.feature_names
                print(f"   [OK] Feature names loaded: {len(self.feature_names)} features")
            else:
                print(f"   [INFO] XGBoost loaded (feature names not stored)")
            
            self.load_label_mapping()
            print(f"   [OK] XGBoost mode: ONLY flag anomalies with 100% confidence (threshold = {self.anomaly_threshold*100:.0f}%)")
            
        except Exception as e:
            print(f"   [ERROR] XGBoost load failed: {e}")
            import traceback
            traceback.print_exc()
    
    def load_label_mapping(self):
        """Load shared label mapping"""
        label_file = 'model/label_mapping.json'
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.label_mapping = json.load(f)
                if all(isinstance(k, str) for k in self.label_mapping.keys()):
                    self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
            print(f"   [OK] Label mapping loaded: {len(self.label_mapping)} classes")
        else:
            print(f"   [WARNING] label_mapping.json not found at {label_file}")
    
    def prepare_features(self, df):
        """Prepare features - use numpy array to avoid feature name issues"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values
        return np.nan_to_num(X)
    
    def predict(self, features_file, output_file=None):
        """Make predictions using XGBoost model - ONLY 100% confidence anomalies"""
        if not os.path.exists(features_file):
            print(f"   [{self.model_name}] No features found at {features_file}")
            return None
        
        df = pd.read_csv(features_file)
        print(f"   [{self.model_name}] Loaded {len(df)} samples")
        
        if not self.model_loaded:
            print(f"   [{self.model_name}] Model not loaded - cannot predict")
            return None
        
        X = self.prepare_features(df)
        if X is None:
            return None
        
        print(f"   [{self.model_name}] Feature shape: {X.shape}")
        
        try:
            predictions_raw = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            print(f"   [{self.model_name}] Prediction error: {e}")
            return None
        
        # Process predictions - ONLY flag at 100% confidence
        results = self._process_predictions_strict(df, predictions_raw, probabilities)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            print(f"   [{self.model_name}] Saved to {os.path.basename(output_file)}")
        
        return results
    
    def _process_predictions_strict(self, df, predictions_raw, probabilities):
        """
        Process predictions - ONLY flag anomalies when confidence is 100%
        """
        predictions_binary = []
        anomaly_probs = []
        prediction_labels = []
        
        normal_class = self._find_normal_class()
        
        for i, prob in enumerate(probabilities):
            # Get normal probability
            if normal_class is not None and normal_class < len(prob):
                normal_prob = prob[normal_class]
            else:
                normal_prob = prob[0] if len(prob) > 0 else 0
            
            anomaly_prob = 1 - normal_prob
            
            # STRICT: ONLY flag as anomaly if confidence is EXACTLY 1.00 (100%)
            # Since 100% confidence is nearly impossible in real-world ML,
            # this will almost always result in is_anomaly = False
            is_anomaly = (anomaly_prob >= self.anomaly_threshold)  # threshold = 1.00
            
            predictions_binary.append(1 if is_anomaly else 0)
            anomaly_probs.append(anomaly_prob)
            
            pred_class = predictions_raw[i]
            if self.label_mapping and pred_class in self.label_mapping:
                attack_type = self.label_mapping[pred_class]
            else:
                attack_type = f"Class {pred_class}"
            
            if is_anomaly:
                prediction_labels.append(f"ANOMALY - {attack_type} (100% confidence)")
            else:
                prediction_labels.append("NORMAL")
        
        df_copy = df.copy()
        df_copy['prediction'] = predictions_binary
        df_copy['anomaly_probability'] = anomaly_probs
        df_copy['prediction_label'] = prediction_labels
        df_copy['model_name'] = self.model_name
        df_copy['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        anomaly_count = sum(predictions_binary)
        anomaly_rate = (anomaly_count / len(df) * 100) if len(df) > 0 else 0
        print(f"   [{self.model_name}] {len(df)-anomaly_count} normal, {anomaly_count} anomalies ({anomaly_rate:.1f}%)")
        print(f"   [{self.model_name}] STRICT MODE: Only flagging anomalies with 100% confidence")
        print(f"   [{self.model_name}] Max anomaly probability: {max(anomaly_probs):.6f}")
        
        if anomaly_count == 0:
            print(f"   [{self.model_name}] No anomalies at 100% confidence threshold")
        
        return df_copy
    
    def _find_normal_class(self):
        """Find which class is Normal/Benign"""
        if self.label_mapping:
            for class_id, label in self.label_mapping.items():
                if 'normal' in str(label).lower() or 'benign' in str(label).lower():
                    return class_id
        return 0