# src/rule_based_predictor.py - Complete Rule-Based Detection with Duration & Density Indicators
import os
import pandas as pd
import numpy as np
from collections import Counter

class RuleBasedPredictor:
    def __init__(self, threshold=0.5):
        """
        Initialize rule-based predictor with duration and density indicators
        
        Args:
            threshold: Anomaly threshold (0-1) - higher = fewer detections
        """
        self.threshold = threshold
        print("   Using enhanced rule-based detection with duration & density indicators")
    
    def predict(self, flows_path, output_path='data/rule_predictions.csv'):
        """
        Detect anomalies using comprehensive rules including:
        - Long sessions
        - Short repetitive bursts
        - Slow drip attacks
        - High density flows
        - High packet/byte rates
        """
        print(f"   Loading flows from: {flows_path}")
        df = pd.read_csv(flows_path)
        
        # Initialize
        df['anomaly_score'] = 0
        df['prediction'] = 0
        df['anomaly_reason'] = ''
        
        # ============================================================
        # EXISTING RULES (Keep these)
        # ============================================================
        
        # Rule 1: High packet rate (above 95th percentile)
        if 'packet_rate' in df.columns and len(df) > 5:
            pkt_thresh = df['packet_rate'].quantile(0.95)
            mask = df['packet_rate'] > pkt_thresh
            df.loc[mask, 'anomaly_score'] += 0.4
            df.loc[mask, 'anomaly_reason'] += 'High packet rate; '
            print(f"      Packet rate threshold: {pkt_thresh:.2f} pps")
        
        # Rule 2: High byte rate (above 95th percentile)
        if 'byte_rate' in df.columns and len(df) > 5:
            byte_thresh = df['byte_rate'].quantile(0.95)
            mask = df['byte_rate'] > byte_thresh
            df.loc[mask, 'anomaly_score'] += 0.4
            df.loc[mask, 'anomaly_reason'] += 'High byte rate; '
            print(f"      Byte rate threshold: {byte_thresh:.2f} Bps")
        
        # Rule 3: Unusual packet size (above 95th or below 5th percentile)
        if 'avg_packet_size' in df.columns and len(df) > 5:
            size_high = df['avg_packet_size'].quantile(0.95)
            size_low = df['avg_packet_size'].quantile(0.05)
            mask_high = df['avg_packet_size'] > size_high
            mask_low = df['avg_packet_size'] < size_low
            df.loc[mask_high, 'anomaly_score'] += 0.3
            df.loc[mask_low, 'anomaly_score'] += 0.3
            df.loc[mask_high, 'anomaly_reason'] += 'Unusually large packets; '
            df.loc[mask_low, 'anomaly_reason'] += 'Unusually small packets; '
            print(f"      Packet size range: {size_low:.0f} - {size_high:.0f} bytes")
        
        # Rule 4: Long duration sessions (above 95th percentile)
        if 'duration' in df.columns and len(df) > 5:
            dur_thresh = df['duration'].quantile(0.95)
            mask = df['duration'] > dur_thresh
            df.loc[mask, 'anomaly_score'] += 0.3
            df.loc[mask, 'anomaly_reason'] += 'Long duration session; '
            print(f"      Duration threshold: {dur_thresh:.2f} seconds")
        
        # Rule 5: High packet density (packets per second)
        if 'packets' in df.columns and 'duration' in df.columns:
            density = df['packets'] / df['duration']
            density_thresh = density.quantile(0.95) if len(df) > 5 else 0
            mask = density > density_thresh
            df.loc[mask, 'anomaly_score'] += 0.4
            df.loc[mask, 'anomaly_reason'] += 'High packet density; '
            print(f"      Packet density threshold: {density_thresh:.2f} pps")
        
        # ============================================================
        # NEW RULES - Duration and Traffic Density Indicators
        # ============================================================
        
        # Rule 6: Short bursts (very short duration but high packet rate)
        if 'duration' in df.columns and 'packet_rate' in df.columns and len(df) > 5:
            short_duration = df['duration'] < df['duration'].quantile(0.1)
            high_packet_rate = df['packet_rate'] > df['packet_rate'].quantile(0.9)
            short_burst = short_duration & high_packet_rate
            df.loc[short_burst, 'anomaly_score'] += 0.5
            df.loc[short_burst, 'anomaly_reason'] += 'Short burst (many packets in short time); '
            if short_burst.any():
                print(f"      Short bursts detected: {short_burst.sum()} flows")
        
        # Rule 7: Slow drip (long duration + low byte rate)
        if 'duration' in df.columns and 'byte_rate' in df.columns and len(df) > 5:
            long_duration = df['duration'] > df['duration'].quantile(0.9)
            low_byte_rate = df['byte_rate'] < df['byte_rate'].quantile(0.1)
            slow_drip = long_duration & low_byte_rate
            df.loc[slow_drip, 'anomaly_score'] += 0.5
            df.loc[slow_drip, 'anomaly_reason'] += 'Slow drip (long session with low traffic); '
            if slow_drip.any():
                print(f"      Slow drip attacks detected: {slow_drip.sum()} flows")
        
        # Rule 8: Long dense flows (long duration + high packet density)
        if 'duration' in df.columns and len(df) > 5:
            long_duration = df['duration'] > df['duration'].quantile(0.9)
            high_density = density > density_thresh if 'density' in locals() else False
            long_dense = long_duration & high_density
            df.loc[long_dense, 'anomaly_score'] += 0.6
            df.loc[long_dense, 'anomaly_reason'] += 'Long dense flow (sustained high activity); '
            if long_dense.any():
                print(f"      Long dense flows detected: {long_dense.sum()} flows")
        
        # Rule 9: Short repetitive bursts (same IP with many short flows)
        if 'src_ip' in df.columns and 'duration' in df.columns:
            # Identify short flows (duration < 10th percentile)
            short_flows = df[df['duration'] < df['duration'].quantile(0.1)]
            if len(short_flows) > 0:
                # Count bursts per source IP
                burst_counts = short_flows['src_ip'].value_counts()
                # Find IPs with excessive bursts (> 90th percentile)
                burst_threshold = burst_counts.quantile(0.9) if len(burst_counts) > 5 else 5
                suspicious_ips = burst_counts[burst_counts > burst_threshold].index
                # Flag flows from these IPs that are short
                mask = df['src_ip'].isin(suspicious_ips) & (df['duration'] < df['duration'].quantile(0.1))
                df.loc[mask, 'anomaly_score'] += 0.5
                df.loc[mask, 'anomaly_reason'] += 'Repetitive short bursts (possible scanning); '
                if mask.any():
                    print(f"      Repetitive bursts detected: {mask.sum()} flows from {len(suspicious_ips)} IPs")
        
        # Rule 10: Very short duration with many packets (possible scan/flood)
        if 'packets' in df.columns and 'duration' in df.columns:
            very_short = df['duration'] < df['duration'].quantile(0.05)
            many_packets = df['packets'] > df['packets'].quantile(0.9)
            scan_activity = very_short & many_packets
            df.loc[scan_activity, 'anomaly_score'] += 0.5
            df.loc[scan_activity, 'anomaly_reason'] += 'Possible scan/flood (many packets in short time); '
            if scan_activity.any():
                print(f"      Scan/flood activity detected: {scan_activity.sum()} flows")
        
        # Rule 11: Very low packet density (unusual silence or keep-alive)
        if 'packet_rate' in df.columns and len(df) > 5:
            low_density = df['packet_rate'] < df['packet_rate'].quantile(0.05)
            df.loc[low_density, 'anomaly_score'] += 0.3
            df.loc[low_density, 'anomaly_reason'] += 'Very low packet density (unusual silence); '
            if low_density.any():
                print(f"      Very low density flows detected: {low_density.sum()} flows")
        
        # ============================================================
        # COMBINED DURATION + DENSITY SCORING
        # ============================================================
        
        # Rule 12: Duration + Rate combined (long + high byte rate)
        if 'duration' in df.columns and 'byte_rate' in df.columns and len(df) > 5:
            long_session = df['duration'] > df['duration'].quantile(0.9)
            high_byte = df['byte_rate'] > df['byte_rate'].quantile(0.9)
            data_exfil = long_session & high_byte
            df.loc[data_exfil, 'anomaly_score'] += 0.5
            df.loc[data_exfil, 'anomaly_reason'] += 'Long session with high byte rate (possible data exfiltration); '
            if data_exfil.any():
                print(f"      Data exfiltration patterns detected: {data_exfil.sum()} flows")
        
        # Rule 13: Duration + Density combined (long + high packet density)
        if 'duration' in df.columns and 'packet_rate' in df.columns and len(df) > 5:
            long_session = df['duration'] > df['duration'].quantile(0.9)
            high_density_rate = df['packet_rate'] > df['packet_rate'].quantile(0.9)
            persistent_attack = long_session & high_density_rate
            df.loc[persistent_attack, 'anomaly_score'] += 0.5
            df.loc[persistent_attack, 'anomaly_reason'] += 'Persistent attack pattern (long + high density); '
            if persistent_attack.any():
                print(f"      Persistent attack patterns detected: {persistent_attack.sum()} flows")
        
        # ============================================================
        # FINAL SCORING AND OUTPUT
        # ============================================================
        
        # Clip score to 0-1 range
        df['anomaly_score'] = df['anomaly_score'].clip(0, 1)
        
        # Determine prediction based on threshold
        df['prediction'] = (df['anomaly_score'] > self.threshold).astype(int)
        df['prediction_label'] = df['prediction'].apply(lambda x: 'ANOMALY' if x == 1 else 'NORMAL')
        df['anomaly_probability'] = df['anomaly_score']
        
        # Clean up anomaly reason (remove trailing semicolon)
        df['anomaly_reason'] = df['anomaly_reason'].str.rstrip('; ')
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Summary
        anomalies = df['prediction'].sum()
        print(f"\n   " + "="*50)
        print(f"   PREDICTION SUMMARY")
        print(f"   " + "="*50)
        print(f"      Total flows: {len(df)}")
        print(f"      Normal: {len(df) - anomalies}")
        print(f"      Anomalies: {anomalies}")
        print(f"      Anomaly rate: {anomalies/len(df)*100:.1f}%")
        
        # Show top anomaly reasons
        if anomalies > 0:
            print(f"\n   Top Anomaly Reasons:")
            reasons = df[df['prediction'] == 1]['anomaly_reason'].value_counts().head(5)
            for reason, count in reasons.items():
                if reason:
                    print(f"      • {reason}: {count} flows")
        
        return df


def main():
    flows_file = 'data/processed_flows/flows.csv'
    if not os.path.exists(flows_file):
        print("No flows file found")
        return
    
    predictor = RuleBasedPredictor(threshold=0.5)
    df = predictor.predict(flows_file)
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    cols = ['src_ip', 'dst_ip', 'packets', 'packet_rate', 'duration', 
            'prediction_label', 'anomaly_score', 'anomaly_reason']
    available = [c for c in cols if c in df.columns]
    print(df[available].head(15).to_string())


if __name__ == "__main__":
    main()