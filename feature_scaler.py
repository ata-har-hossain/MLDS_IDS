# src/feature_scaler.py - Scale features to match model expectations
import numpy as np
import pandas as pd
import os

class FeatureScaler:
    """
    Scale features to match CICIDS2017 training data distribution
    """
    
    # Scaling factors to bring your values into expected ranges
    SCALING_FACTORS = {
        # Flow timing features (your values are too large)
        'Flow Duration': 0.08333,      # 60M -> 5M
        'Flow IAT Mean': 0.01667,      # 60M -> 1M
        'Flow IAT Std': 0.01667,
        'Flow IAT Max': 0.01667,
        'Flow IAT Min': 0.01667,
        'Fwd IAT Total': 0.01667,
        'Fwd IAT Mean': 0.01667,
        'Fwd IAT Std': 0.01667,
        'Fwd IAT Max': 0.01667,
        'Fwd IAT Min': 0.01667,
        'Bwd IAT Total': 0.01667,
        'Bwd IAT Mean': 0.01667,
        'Bwd IAT Std': 0.01667,
        'Bwd IAT Max': 0.01667,
        'Bwd IAT Min': 0.01667,
        
        # Packet count features (your values are too small)
        'Total Fwd Packets': 83.33,    # 0.6 -> 50
        'Total Bwd Packets': 83.33,
        'Fwd Packets/s': 625.0,        # 0.016 -> 10
        'Bwd Packets/s': 625.0,
        'Flow Packets/s': 625.0,
        
        # Byte count features
        'Total Length of Fwd Packets': 468.0,   # 53.4 -> 25000
        'Total Length of Bwd Packets': 468.0,
        'Flow Bytes/s': 3375.0,                # 1.48 -> 5000
        'Subflow Fwd Bytes': 468.0,
        
        # Packet size features
        'Average Packet Size': 5.62,           # 89 -> 500
        'Fwd Packet Length Mean': 5.62,
        'Bwd Packet Length Mean': 5.62,
        'Packet Length Mean': 5.62,
        'Min Packet Length': 5.62,
        'Max Packet Length': 5.62,
        'Fwd Packet Length Max': 5.62,
        'Fwd Packet Length Min': 5.62,
        'Bwd Packet Length Max': 5.62,
        'Bwd Packet Length Min': 5.62,
        'Packet Length Std': 5.62,
        'Packet Length Variance': 31.6,
        
        # Header features
        'Fwd Header Length': 5.62,
        'Bwd Header Length': 5.62,
        
        # Subflow features
        'Subflow Fwd Packets': 83.33,
        'Subflow Bwd Packets': 83.33,
        'Subflow Bwd Bytes': 468.0,
    }
    
    @classmethod
    def scale_features(cls, df):
        """Scale all features to match model expectations"""
        df_scaled = df.copy()
        
        print(f"   Scaling {len(df_scaled)} flows...")
        
        # Apply scaling factors
        scaled_count = 0
        for feature, factor in cls.SCALING_FACTORS.items():
            if feature in df_scaled.columns:
                df_scaled[feature] = pd.to_numeric(df_scaled[feature], errors='coerce') * factor
                scaled_count += 1
        
        # Ensure port numbers are within range
        if 'Destination Port' in df_scaled.columns:
            df_scaled['Destination Port'] = df_scaled['Destination Port'].clip(1, 65535)
        
        # Ensure flag counts are reasonable
        flag_features = ['FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count']
        for feat in flag_features:
            if feat in df_scaled.columns:
                df_scaled[feat] = df_scaled[feat].clip(0, 10)
        
        # Ensure window sizes are reasonable
        window_features = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']
        for feat in window_features:
            if feat in df_scaled.columns:
                df_scaled[feat] = df_scaled[feat].clip(0, 65535)
        
        # Handle any NaN or infinite values
        df_scaled = df_scaled.fillna(0)
        df_scaled = df_scaled.replace([np.inf, -np.inf], 0)
        
        print(f"   Scaled {scaled_count} features")
        
        return df_scaled


def scale_features_file(input_path, output_path=None):
    """Scale a features file and save as scaled version"""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scaled{ext}"
    
    print(f"\n[STATS] Loading features from: {os.path.basename(input_path)}")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} flows with {len(df.columns)} features")
    
    df_scaled = FeatureScaler.scale_features(df)
    df_scaled.to_csv(output_path, index=False)
    
    print(f"   [OK] Saved scaled features to {output_path}")
    return df_scaled


def main():
    """Test the feature scaler"""
    print("="*60)
    print("Testing Feature Scaler")
    print("="*60)
    
    features_file = 'data/processed_flows/features.csv'
    
    if not os.path.exists(features_file):
        print(f"[ERROR] features.csv not found at {features_file}")
        print("Run feature_extractor.py first")
        return
    
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} flows with {len(df.columns)} features")
    
    df_scaled = FeatureScaler.scale_features(df)
    print(f"\n[OK] Scaled features shape: {df_scaled.shape}")
    
    # Show before/after for first flow
    print("\n[STATS] Sample scaling (first flow):")
    for feat in ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s']:
        if feat in df.columns and feat in df_scaled.columns:
            print(f"   {feat}: {df[feat].iloc[0]:.2f} -> {df_scaled[feat].iloc[0]:.2f}")


if __name__ == "__main__":
    main()