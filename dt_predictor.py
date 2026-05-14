# src/dt_predictor.py - Decision Tree Predictor (100% thresholds)
import joblib
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

class DT_Predictor:
    def __init__(self, model_path='model/dt_model.pkl', anomaly_threshold=1.00):
        """
        Initialize Decision Tree Predictor with 100% confidence thresholds
        
        Args:
            model_path: Path to model file (dt_model.pkl)
            anomaly_threshold: Threshold for anomaly detection (1.00 = 100%)
        """
        self.model_path = model_path
        self.model_name = 'Decision Tree'
        self.model = None
        self.anomaly_threshold = anomaly_threshold
        self.model_loaded = False
        self.label_mapping = None
        
        # ENSURE ALL THRESHOLDS ARE 1.00 (100%)
        self.normal_confidence_threshold = 1.00    # 100% confidence needed for NORMAL
        self.attack_confidence_threshold = 1.00    # 100% confidence needed for ATTACK
        
        self.load_model()
    
    def load_model(self):
        """Load the Decision Tree model from pickle file"""
        try:
            if not os.path.exists(self.model_path):
                print(f"   [WARNING] Decision Tree model not found at {self.model_path}")
                return
            
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            
            # CRITICAL FIX: Disable feature name validation
            if hasattr(self.model, 'feature_names_in_'):
                self.model.feature_names_in_ = None
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
                print(f"   [OK] Decision Tree loaded with {len(self.feature_names)} features")
            else:
                print(f"   [OK] Decision Tree loaded")
            
            self.load_label_mapping()
            print(f"   [OK] STRICT MODE: Normal=100%, Attack=100%, Fallback=100%")
            print(f"   [OK] All thresholds set to 100% - only flags anomalies at 100% confidence")
            
        except Exception as e:
            print(f"   [ERROR] Decision Tree load failed: {e}")
    
    def load_label_mapping(self):
        """Load shared label mapping"""
        label_file = 'model/label_mapping.json'
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.label_mapping = json.load(f)
                if all(isinstance(k, str) for k in self.label_mapping.keys()):
                    self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
            print(f"   [OK] Label mapping loaded: {len(self.label_mapping)} classes")
    
    def prepare_features(self, df):
        """Convert to numpy array to avoid feature name issues"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values
        return np.nan_to_num(X)
    
    def predict(self, features_file, output_file=None):
        """Make predictions using Decision Tree model with 100% thresholds"""
        if not os.path.exists(features_file):
            print(f"   [{self.model_name}] No features found at {features_file}")
            return None
        
        df = pd.read_csv(features_file)
        print(f"   [{self.model_name}] Loaded {len(df)} samples")
        
        if not self.model_loaded:
            print(f"   [{self.model_name}] Model not loaded")
            return None
        
        # Prepare features as numpy array
        X = self.prepare_features(df)
        if X is None:
            return None
        
        print(f"   [{self.model_name}] Feature shape: {X.shape}")
        
        try:
            predictions_raw = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            print(f"   [{self.model_name}] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Process predictions with STRICT 100% thresholds
        results = self._process_predictions_strict(df, predictions_raw, probabilities)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results.to_csv(output_file, index=False)
            print(f"   [{self.model_name}] Saved to {os.path.basename(output_file)}")
        
        return results
    
    def _process_predictions_strict(self, df, predictions_raw, probabilities):
        """
        Process predictions using STRICT mode with 100% confidence thresholds.
        
        Strict Mode Logic (all thresholds = 1.00):
        - A flow is classified as NORMAL ONLY if NORMAL probability = 1.00 (100%)
        - A flow is classified as ANOMALY ONLY if MAX_ATTACK probability = 1.00 (100%)
        - Since 100% confidence is nearly impossible, most flows will use the 
          fallback anomaly_threshold (which is also 1.00)
        - With anomaly_threshold=1.00, EVERYTHING is classified as NORMAL
        """
        predictions_binary = []
        anomaly_probs = []
        prediction_labels = []
        
        # Find normal class (usually 0)
        normal_class = self._find_normal_class()
        
        for i, prob in enumerate(probabilities):
            # Get normal probability
            if normal_class is not None and normal_class < len(prob):
                normal_prob = prob[normal_class]
            else:
                normal_prob = prob[0] if len(prob) > 0 else 0
            
            # Get maximum attack probability
            attack_probs = [p for idx, p in enumerate(prob) if idx != normal_class]
            max_attack_prob = max(attack_probs) if attack_probs else 0
            
            # STRICT MODE DECISION LOGIC (all thresholds = 1.00)
            is_anomaly = False
            
            # Case 1: 100% confidence NORMAL
            if normal_prob >= self.normal_confidence_threshold:
                is_anomaly = False
            
            # Case 2: 100% confidence ATTACK
            elif max_attack_prob >= self.attack_confidence_threshold:
                is_anomaly = True
            
            # Case 3: Not 100% confident - use fallback threshold (which is also 1.00)
            else:
                # With anomaly_threshold=1.00, this will ALWAYS be False
                is_anomaly = (1 - normal_prob) >= self.anomaly_threshold
            
            predictions_binary.append(1 if is_anomaly else 0)
            anomaly_probs.append(1 - normal_prob)
            
            # Get attack type label
            pred_class = predictions_raw[i]
            if self.label_mapping and pred_class in self.label_mapping:
                attack_type = self.label_mapping[pred_class]
            else:
                attack_type = f"Class {pred_class}"
            
            if is_anomaly:
                prediction_labels.append(f"ANOMALY - {attack_type}")
            else:
                prediction_labels.append("NORMAL")
        
        df_copy = df.copy()
        df_copy['prediction'] = predictions_binary
        df_copy['anomaly_probability'] = anomaly_probs
        df_copy['prediction_label'] = prediction_labels
        df_copy['model_name'] = self.model_name
        df_copy['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add threshold info as metadata
        df_copy['normal_threshold'] = self.normal_confidence_threshold
        df_copy['attack_threshold'] = self.attack_confidence_threshold
        df_copy['anomaly_threshold'] = self.anomaly_threshold
        
        anomaly_count = sum(predictions_binary)
        anomaly_rate = (anomaly_count / len(df) * 100) if len(df) > 0 else 0
        print(f"   [{self.model_name}] STRICT MODE: Normal=100%, Attack=100%, Fallback=100%")
        print(f"   [{self.model_name}] {len(df)-anomaly_count} normal, {anomaly_count} anomalies ({anomaly_rate:.1f}%)")
        
        return df_copy
    
    def _find_normal_class(self):
        """Find which class is Normal/Benign"""
        if self.label_mapping:
            for class_id, label in self.label_mapping.items():
                if 'normal' in str(label).lower() or 'benign' in str(label).lower():
                    return class_id
        return 0