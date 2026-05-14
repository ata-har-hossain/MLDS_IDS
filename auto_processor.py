# auto_processor.py - Updated with only RF, XGB, ET, DT, and GB
import os
import sys
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from flow_generator import FlowGenerator
from rule_based_predictor import RuleBasedPredictor
from feature_extractor import FeatureExtractor
from feature_scaler import FeatureScaler

# Import only the working predictors
from rf_predictor import RF_Predictor
from xgb_predictor import XGB_Predictor
from et_predictor import ET_Predictor
from dt_predictor import DT_Predictor      # Decision Tree
from gb_predictor import GB_Predictor      # Gradient Boosting


class AutoPCAPProcessor:
    def __init__(self, pcap_dir='data/raw_pcaps', flows_dir='data/processed_flows'):
        self.pcap_dir = pcap_dir
        self.flows_dir = flows_dir
        self.processed_files = set()
        
        # Create directories
        os.makedirs(pcap_dir, exist_ok=True)
        os.makedirs(flows_dir, exist_ok=True)
        
        # Initialize predictors
        print("\n" + "="*60)
        print("LOADING ML MODELS")
        print("="*60)
        
        self.predictors = {
            'rf': RF_Predictor('model/rf_model.pkl', anomaly_threshold=1.00),
            'xgb': XGB_Predictor('model/xgb_model.json', anomaly_threshold=0.99),
            'et': ET_Predictor('model/et_model.pkl', anomaly_threshold=1.00),
            'dt': DT_Predictor('model/dt_model.pkl', anomaly_threshold=1.00),
            'gb': GB_Predictor('model/gb_model.joblib', anomaly_threshold=1.00)
        }
        
        # Track which models successfully loaded
        self.active_predictors = {
            name: pred for name, pred in self.predictors.items() if pred.model_loaded
        }
        
        print("\n" + "-"*40)
        print(f"Active models: {len(self.active_predictors)}/5")
        for name in self.active_predictors.keys():
            print(f"  - {name.upper()}")
        print("-"*40 + "\n")
        
        self.load_processed_history()
    
    def load_processed_history(self):
        """Load previously processed files"""
        history_file = f'{self.flows_dir}/processed.txt'
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.processed_files = set(line.strip() for line in f)
                print(f"Loaded {len(self.processed_files)} previously processed files\n")
            except:
                pass
    
    def save_processed_history(self):
        """Save processed files"""
        history_file = f'{self.flows_dir}/processed.txt'
        try:
            with open(history_file, 'w') as f:
                for filename in self.processed_files:
                    f.write(f"{filename}\n")
        except:
            pass
    
    def process_pcap(self, pcap_file):
        """Process a single PCAP file with all models"""
        filename = os.path.basename(pcap_file)
        
        if filename in self.processed_files:
            return False
        
        if os.path.getsize(pcap_file) == 0:
            print(f"\n[SKIP] {filename} - empty file")
            return False
        
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing: {filename}")
        print(f"   File size: {os.path.getsize(pcap_file) / 1024:.1f} KB")
        print(f"{'='*60}")
        
        # STEP 1: Generate flows
        flows_output = f'{self.flows_dir}/flows.csv'
        try:
            generator = FlowGenerator(pcap_file, flows_output)
            df_flows = generator.run()
            if df_flows is None or len(df_flows) == 0:
                print(f"   [ERROR] No flows extracted")
                return False
            print(f"   [OK] Generated {len(df_flows)} flows")
        except Exception as e:
            print(f"   [ERROR] Flow generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # STEP 2: Extract 52 features
        try:
            extractor = FeatureExtractor(flows_output)
            X_features, _ = extractor.process()
            features_output = f'{self.flows_dir}/features.csv'
            X_features.to_csv(features_output, index=False)
            print(f"   [OK] Extracted {X_features.shape[1]} features")
        except Exception as e:
            print(f"   [ERROR] Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # STEP 3: Scale features
        try:
            X_scaled = FeatureScaler.scale_features(X_features)
            scaled_output = f'{self.flows_dir}/features_scaled.csv'
            X_scaled.to_csv(scaled_output, index=False)
            print(f"   [OK] Features scaled")
        except Exception as e:
            print(f"   [WARNING] Scaling error: {e}, using raw features")
            scaled_output = features_output
        
        # STEP 4: Rule-based detection
        try:
            rule_predictor = RuleBasedPredictor(threshold=0.5)
            rule_output = f'{self.flows_dir}/rule_predictions.csv'
            df_rule = rule_predictor.predict(flows_output, rule_output)
            rule_anomalies = df_rule['prediction'].sum() if df_rule is not None else 0
            print(f"   [RULE] {rule_anomalies} anomalies")
        except Exception as e:
            print(f"   [WARNING] Rule-based error: {e}")
        
        # STEP 5: Run all ML models
        ml_results = {}
        print(f"\n   Running {len(self.active_predictors)} ML models...")
        print("-" * 40)
        
        for name, predictor in self.active_predictors.items():
            try:
                output_file = f'{self.flows_dir}/{name}_predictions.csv'
                print(f"   [{name.upper()}] Predicting...")
                
                df_result = predictor.predict(scaled_output, output_file)
                
                if df_result is not None and len(df_result) > 0:
                    ml_results[name] = df_result
                    anomalies = df_result['prediction'].sum()
                    print(f"   [OK] [{name.upper()}] {anomalies} anomalies detected")
                    
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"      File: {name}_predictions.csv ({file_size} bytes)")
                    else:
                        print(f"      WARNING: Output file not found!")
                else:
                    print(f"   [ERROR] [{name.upper()}] Returned empty result")
                    
            except Exception as e:
                print(f"   [ERROR] [{name.upper()}] Failed: {str(e)[:100]}")
        
        print("-" * 40)
        print(f"   ML Models Successful: {len(ml_results)}/{len(self.active_predictors)}")
        for name in ml_results.keys():
            print(f"      [OK] {name.upper()}")
        if len(ml_results) < len(self.active_predictors):
            failed = set(self.active_predictors.keys()) - set(ml_results.keys())
            print(f"      [FAILED] {', '.join(failed).upper()}")
        
        # STEP 6: Ensemble prediction
        if len(ml_results) >= 2:
            self._create_ensemble_predictions(ml_results)
        
        # STEP 7: Save summary
        self._save_processing_summary(filename, df_flows, rule_anomalies, ml_results)
        
        self.processed_files.add(filename)
        self.save_processed_history()
        
        print(f"\n{'='*60}")
        print(f"   Processing complete for {filename}")
        print(f"   Total flows: {len(df_flows)}")
        print(f"   Active models: {len(ml_results)}/{len(self.active_predictors)}")
        print(f"{'='*60}")
        
        return True
    
    def _create_ensemble_predictions(self, ml_results):
        """Create ensemble using majority voting"""
        print(f"\n   [ENSEMBLE] Creating Ensemble Prediction (Majority Vote)...")
        
        predictions_list = []
        for name, df in ml_results.items():
            if 'prediction' in df.columns:
                predictions_list.append(df['prediction'].values)
        
        if len(predictions_list) >= 2:
            predictions_array = np.array(predictions_list)
            ensemble_pred = (predictions_array.sum(axis=0) >= (len(predictions_list) / 2)).astype(int)
            ensemble_probs = predictions_array.mean(axis=0)
            
            first_model = list(ml_results.values())[0]
            ensemble_df = first_model.copy()
            ensemble_df['prediction'] = ensemble_pred
            ensemble_df['anomaly_probability'] = ensemble_probs
            ensemble_df['model_name'] = 'Ensemble (Majority Vote)'
            ensemble_df['prediction_label'] = ensemble_df['prediction'].apply(
                lambda x: 'ANOMALY' if x == 1 else 'NORMAL'
            )
            
            ensemble_output = f'{self.flows_dir}/ensemble_predictions.csv'
            ensemble_df.to_csv(ensemble_output, index=False)
            
            ensemble_anomalies = ensemble_pred.sum()
            print(f"   [OK] [ENSEMBLE] {ensemble_anomalies} anomalies (majority vote of {len(predictions_list)} models)")
    
    def _save_processing_summary(self, filename, df_flows, rule_anomalies, ml_results):
        """Save processing summary to CSV"""
        try:
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pcap_file': filename,
                'total_flows': len(df_flows),
                'rule_anomalies': rule_anomalies,
                'rule_rate': f"{(rule_anomalies/len(df_flows)*100):.1f}%" if len(df_flows) > 0 else "0%"
            }
            
            for name, df in ml_results.items():
                anomalies = df['prediction'].sum()
                summary[f'{name}_anomalies'] = anomalies
                summary[f'{name}_rate'] = f"{(anomalies/len(df_flows)*100):.1f}%" if len(df_flows) > 0 else "0%"
            
            summary_file = f'{self.flows_dir}/processing_summary.csv'
            summary_df = pd.DataFrame([summary])
            if os.path.exists(summary_file):
                existing = pd.read_csv(summary_file)
                summary_df = pd.concat([existing, summary_df], ignore_index=True)
            summary_df.to_csv(summary_file, index=False)
        except Exception as e:
            print(f"   [WARNING] Could not save summary: {e}")
    
    def check_and_process(self):
        """Check for new PCAP files and process them"""
        pcap_files = glob.glob(f'{self.pcap_dir}/*.pcap')
        
        if not pcap_files:
            return False
        
        pcap_files.sort(key=os.path.getctime)
        
        processed_any = False
        for pcap_file in pcap_files:
            filename = os.path.basename(pcap_file)
            if filename not in self.processed_files:
                file_age = time.time() - os.path.getctime(pcap_file)
                if file_age >= 5:
                    if self.process_pcap(pcap_file):
                        processed_any = True
                else:
                    print(f"\n[WAIT] {filename} is still being written (age: {file_age:.0f}s)")
        
        return processed_any
    
    def run_continuous(self, check_interval=10):
        """Continuous monitoring for new PCAP files"""
        print("="*60)
        print("AUTO PROCESSOR - Multi-Model IDS")
        print("="*60)
        print(f"Watching: {self.pcap_dir}")
        print(f"Output: {self.flows_dir}")
        print(f"Check interval: {check_interval}s")
        print(f"Active models: {len(self.active_predictors)}/5")
        print("="*60)
        
        print("\nChecking for existing PCAP files...")
        self.check_and_process()
        
        print("\nMonitoring for new PCAP files...")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            while True:
                self.check_and_process()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("Stopping processor...")
            print(f"Total files processed: {len(self.processed_files)}")
            print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto PCAP Processor with 5 ML Models')
    parser.add_argument('--once', action='store_true', help='Process once and exit')
    parser.add_argument('-i', '--interval', type=int, default=10, help='Check interval (seconds)')
    parser.add_argument('-d', '--dir', default='data/raw_pcaps', help='PCAP directory')
    
    args = parser.parse_args()
    
    processor = AutoPCAPProcessor(pcap_dir=args.dir)
    
    if args.once:
        print("Running one-time processing...")
        processor.check_and_process()
        print("\nDone!")
    else:
        processor.run_continuous(args.interval)


if __name__ == "__main__":
    main()