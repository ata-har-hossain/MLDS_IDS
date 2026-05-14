# src/feature_extractor.py - Convert flows.csv to features.csv (52 CICIDS2017 features)
import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_feature_names_from_file():
    """
    Load feature names from feature_names.txt file
    Falls back to hardcoded list if file not found
    """
    feature_file = 'model/feature_names.txt'  # Path to your feature_names.txt
    
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        print(f"   Loaded {len(features)} features from {feature_file}")
        return features
    else:
        print(f"   [WARNING] feature_names.txt not found at {feature_file}")
        print(f"   Using hardcoded feature list")
        return get_hardcoded_features()

def get_hardcoded_features():
    """Fallback hardcoded feature list (52 CICIDS2017 features)"""
    return [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "PSH Flag Count", "ACK Flag Count", "Average Packet Size",
        "Subflow Fwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Max", "Active Min", "Idle Mean", "Idle Max", "Idle Min"
    ]

# Load feature names from file (or use hardcoded fallback)
MODEL_FEATURES = load_feature_names_from_file()
print(f"FEATURE COUNT: {len(MODEL_FEATURES)}")


class FeatureExtractor:
    """
    Extracts 52 CICIDS2017 features from flows.csv
    """
    
    def __init__(self, flows_path):
        """
        Initialize Feature Extractor
        
        Args:
            flows_path: Path to flows.csv file
        """
        self.flows_path = flows_path
        self.df = None
    
    def load_flows(self):
        """Load flows from CSV"""
        self.df = pd.read_csv(self.flows_path)
        print(f"   Loaded {len(self.df)} flows")
        return self.df
    
    def map_to_features(self):
        """
        Map basic flow data to 52 CICIDS2017 features
        This uses intelligent defaults and calculations for missing features
        """
        if self.df is None:
            return None
        
        # Create dataframe with all required features, initialized to 0
        X = pd.DataFrame(0.0, index=self.df.index, columns=MODEL_FEATURES)
        
        for idx, row in self.df.iterrows():
            packets = row.get("packets", 0)
            bytes_val = row.get("bytes", 0)
            duration = row.get("duration", 60)
            pkt_rate = row.get("packet_rate", 0)
            byte_rate = row.get("byte_rate", 0)
            avg_size = row.get("avg_packet_size", 0)
            src_port = row.get("src_port", 0)
            dst_port = row.get("dst_port", 0)
            
            # ============================================================
            # BASIC FEATURE MAPPING
            # ============================================================
            
            # Port
            X.at[idx, "Destination Port"] = float(dst_port) if dst_port else 0
            
            # Duration (convert to microseconds for CICIDS format)
            X.at[idx, "Flow Duration"] = float(duration * 1000000) if duration > 0 else 0
            
            # Packet counts (split 60/40 forward/backward)
            X.at[idx, "Total Fwd Packets"] = float(packets * 0.6)
            
            # Byte counts
            X.at[idx, "Total Length of Fwd Packets"] = float(bytes_val * 0.6)
            
            # Packet rates
            X.at[idx, "Flow Packets/s"] = float(pkt_rate)
            X.at[idx, "Flow Bytes/s"] = float(byte_rate)
            X.at[idx, "Fwd Packets/s"] = float(pkt_rate * 0.6) if pkt_rate > 0 else 0
            X.at[idx, "Bwd Packets/s"] = float(pkt_rate * 0.4) if pkt_rate > 0 else 0
            
            # Packet size features
            X.at[idx, "Average Packet Size"] = float(avg_size) if avg_size > 0 else 0
            
            # Packet length features (derived from avg packet size)
            if avg_size > 0:
                X.at[idx, "Fwd Packet Length Mean"] = float(avg_size)
                X.at[idx, "Bwd Packet Length Mean"] = float(avg_size)
                X.at[idx, "Fwd Packet Length Max"] = float(avg_size * 1.5)
                X.at[idx, "Fwd Packet Length Min"] = float(avg_size * 0.5)
                X.at[idx, "Bwd Packet Length Max"] = float(avg_size * 1.5)
                X.at[idx, "Bwd Packet Length Min"] = float(avg_size * 0.5)
                X.at[idx, "Fwd Packet Length Std"] = float(avg_size * 0.3)
                X.at[idx, "Bwd Packet Length Std"] = float(avg_size * 0.3)
                
                # Min/Max packet length
                X.at[idx, "Min Packet Length"] = float(avg_size * 0.5)
                X.at[idx, "Max Packet Length"] = float(avg_size * 1.5)
                X.at[idx, "Packet Length Mean"] = float(avg_size)
                X.at[idx, "Packet Length Std"] = float(avg_size * 0.3)
                X.at[idx, "Packet Length Variance"] = float((avg_size * 0.3) ** 2)
            else:
                # Default values when avg_size is 0
                X.at[idx, "Fwd Packet Length Mean"] = 0.0
                X.at[idx, "Bwd Packet Length Mean"] = 0.0
                X.at[idx, "Fwd Packet Length Max"] = 0.0
                X.at[idx, "Fwd Packet Length Min"] = 0.0
                X.at[idx, "Bwd Packet Length Max"] = 0.0
                X.at[idx, "Bwd Packet Length Min"] = 0.0
                X.at[idx, "Fwd Packet Length Std"] = 0.0
                X.at[idx, "Bwd Packet Length Std"] = 0.0
                X.at[idx, "Min Packet Length"] = 0.0
                X.at[idx, "Max Packet Length"] = 0.0
                X.at[idx, "Packet Length Mean"] = 0.0
                X.at[idx, "Packet Length Std"] = 0.0
                X.at[idx, "Packet Length Variance"] = 0.0
            
            # ============================================================
            # IAT (Inter-Arrival Time) Estimates
            # ============================================================
            if pkt_rate > 0 and duration > 0:
                iat_us = (1.0 / pkt_rate) * 1000000  # Convert to microseconds
                X.at[idx, "Flow IAT Mean"] = iat_us
                X.at[idx, "Flow IAT Std"] = iat_us * 0.3
                X.at[idx, "Flow IAT Max"] = iat_us * 2
                X.at[idx, "Flow IAT Min"] = iat_us * 0.5
                X.at[idx, "Fwd IAT Total"] = iat_us * packets * 0.6
                X.at[idx, "Fwd IAT Mean"] = iat_us
                X.at[idx, "Fwd IAT Std"] = iat_us * 0.3
                X.at[idx, "Fwd IAT Max"] = iat_us * 2
                X.at[idx, "Fwd IAT Min"] = iat_us * 0.5
                X.at[idx, "Bwd IAT Total"] = iat_us * packets * 0.4
                X.at[idx, "Bwd IAT Mean"] = iat_us
                X.at[idx, "Bwd IAT Std"] = iat_us * 0.3
                X.at[idx, "Bwd IAT Max"] = iat_us * 2
                X.at[idx, "Bwd IAT Min"] = iat_us * 0.5
            else:
                # Default values when pkt_rate is 0
                X.at[idx, "Flow IAT Mean"] = 0.0
                X.at[idx, "Flow IAT Std"] = 0.0
                X.at[idx, "Flow IAT Max"] = 0.0
                X.at[idx, "Flow IAT Min"] = 0.0
                X.at[idx, "Fwd IAT Total"] = 0.0
                X.at[idx, "Fwd IAT Mean"] = 0.0
                X.at[idx, "Fwd IAT Std"] = 0.0
                X.at[idx, "Fwd IAT Max"] = 0.0
                X.at[idx, "Fwd IAT Min"] = 0.0
                X.at[idx, "Bwd IAT Total"] = 0.0
                X.at[idx, "Bwd IAT Mean"] = 0.0
                X.at[idx, "Bwd IAT Std"] = 0.0
                X.at[idx, "Bwd IAT Max"] = 0.0
                X.at[idx, "Bwd IAT Min"] = 0.0
            
            # ============================================================
            # HEADER LENGTHS (Estimate ~40 bytes per TCP/UDP header)
            # ============================================================
            X.at[idx, "Fwd Header Length"] = float(packets * 0.6 * 40)
            X.at[idx, "Bwd Header Length"] = float(packets * 0.4 * 40)
            
            # ============================================================
            # FLAG COUNTS (Default values for normal traffic)
            # ============================================================
            X.at[idx, "FIN Flag Count"] = 0.0
            X.at[idx, "PSH Flag Count"] = 0.0
            X.at[idx, "ACK Flag Count"] = 1.0
            
            # ============================================================
            # SUBFLOW STATISTICS (First 100 packets approximation)
            # ============================================================
            X.at[idx, "Subflow Fwd Bytes"] = float(min(int(bytes_val * 0.6), 10000))
            
            # ============================================================
            # WINDOW SIZES (Standard TCP window)
            # ============================================================
            X.at[idx, "Init_Win_bytes_forward"] = 65535.0
            X.at[idx, "Init_Win_bytes_backward"] = 65535.0
            X.at[idx, "act_data_pkt_fwd"] = 1.0 if packets > 0 else 0.0
            X.at[idx, "min_seg_size_forward"] = 1.0
            
            # ============================================================
            # ACTIVE/IDLE TIMES (Set to 0 for normal flows)
            # ============================================================
            X.at[idx, "Active Mean"] = 0.0
            X.at[idx, "Active Max"] = 0.0
            X.at[idx, "Active Min"] = 0.0
            X.at[idx, "Idle Mean"] = 0.0
            X.at[idx, "Idle Max"] = 0.0
            X.at[idx, "Idle Min"] = 0.0
        
        return X
    
    def validate_features(self, X):
        """
        Validate that all expected features are present and have correct data types
        
        Args:
            X: Feature DataFrame to validate
        
        Returns:
            bool: True if validation passes
        """
        missing_features = [f for f in MODEL_FEATURES if f not in X.columns]
        if missing_features:
            print(f"   [WARNING] Missing features: {missing_features[:5]}...")
            return False
        
        # Check for NaN or infinite values
        if X.isna().any().any():
            print(f"   [WARNING] NaN values detected, filling with 0")
            X = X.fillna(0)
        
        if np.isinf(X).any().any():
            print(f"   [WARNING] Infinite values detected, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        return True
    
    def process(self):
        """Complete feature extraction pipeline"""
        self.load_flows()
        X = self.map_to_features()
        
        # Validate features
        self.validate_features(X)
        
        # Add metadata
        if 'pcap_source' in self.df.columns:
            X['pcap_source'] = self.df['pcap_source']
        else:
            X['pcap_source'] = 'unknown'
        
        X['capture_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns: features first, then metadata
        feature_cols = [c for c in X.columns if c not in ['pcap_source', 'capture_time']]
        X = X[feature_cols + ['pcap_source', 'capture_time']]
        
        print(f"   Extracted {len(feature_cols)} features (expected: {len(MODEL_FEATURES)})")
        print(f"   Feature columns: {feature_cols[:5]}...")
        
        # Verify feature count matches model expectation
        if len(feature_cols) != len(MODEL_FEATURES):
            print(f"   [WARNING] Feature count mismatch! Got {len(feature_cols)}, expected {len(MODEL_FEATURES)}")
        
        return X, self.df


def extract_features_from_csv(flows_path, output_path='data/processed_flows/features.csv'):
    """
    Extract features from flows.csv and save to features.csv
    
    Args:
        flows_path: Path to flows.csv
        output_path: Path to save features.csv
    
    Returns:
        DataFrame with features
    """
    print(f"\n[STATS] Extracting features from: {os.path.basename(flows_path)}")
    print("-" * 40)
    
    extractor = FeatureExtractor(flows_path)
    X, df_flows = extractor.process()
    
    # Save features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X.to_csv(output_path, index=False)
    
    print(f"   [OK] Saved {len(X)} flows with {len(X.columns)-2} features to {output_path}")
    
    # Print feature statistics
    print(f"\n   Feature Statistics (first 5 flows):")
    for feat in MODEL_FEATURES[:5]:
        if feat in X.columns:
            non_zero = (X[feat] > 0).sum()
            print(f"      {feat}: {non_zero}/{len(X)} non-zero values")
    
    return X


def verify_features_against_model(model_path='model/rf_model.pkl'):
    """
    Verify that extracted features match what the model expects
    
    Args:
        model_path: Path to the trained model
    """
    try:
        import joblib
        model = joblib.load(model_path)
        
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
            
            print("\n" + "="*60)
            print("Feature Verification against Model")
            print("="*60)
            print(f"Model expects: {len(model_features)} features")
            print(f"Extractor provides: {len(MODEL_FEATURES)} features")
            
            # Check for mismatches
            missing_in_extractor = [f for f in model_features if f not in MODEL_FEATURES]
            extra_in_extractor = [f for f in MODEL_FEATURES if f not in model_features]
            
            if missing_in_extractor:
                print(f"\n❌ Features missing from extractor: {missing_in_extractor[:5]}")
            if extra_in_extractor:
                print(f"\n⚠️ Extra features in extractor: {extra_in_extractor[:5]}")
            
            if not missing_in_extractor and not extra_in_extractor:
                print("\n✅ Feature lists match! Extractor is compatible with model.")
            else:
                print("\n⚠️ Feature mismatch detected. Update MODEL_FEATURES to match model.")
                
    except Exception as e:
        print(f"Could not verify against model: {e}")


def main():
    """Test the feature extractor"""
    print("="*60)
    print("Testing Feature Extractor")
    print("="*60)
    print(f"Features count: {len(MODEL_FEATURES)}")
    print(f"Source: {'feature_names.txt' if os.path.exists('model/feature_names.txt') else 'hardcoded list'}")
    
    flows_file = 'data/processed_flows/flows.csv'
    
    if not os.path.exists(flows_file):
        print(f"\n[ERROR] flows.csv not found at {flows_file}")
        print("Run flow_generator.py first to create flows.csv")
        return
    
    print(f"\n[TEST] Testing with: {flows_file}")
    X = extract_features_from_csv(flows_file)
    
    if X is not None:
        print(f"\n[OK] Success! Features shape: {X.shape}")
        print(f"[OK] Feature columns (first 10): {X.columns[:10].tolist()}")
        print(f"\n[STATS] Sample features (first flow):")
        for feat in MODEL_FEATURES[:10]:
            if feat in X.columns:
                print(f"   {feat}: {X.iloc[0][feat]:.2f}")
        
        # Verify all features are present
        missing = [f for f in MODEL_FEATURES if f not in X.columns]
        if missing:
            print(f"\n[WARNING] Missing features in output: {missing}")
        else:
            print(f"\n[OK] All {len(MODEL_FEATURES)} features present in output")
        
        # Verify against model
        verify_features_against_model()


if __name__ == "__main__":
    main()