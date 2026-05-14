# src/flow_generator.py - TShark-based Flow Generator with Full Output
import os
import subprocess
import pandas as pd
from collections import defaultdict
from datetime import datetime

class FlowGenerator:
    def __init__(self, pcap_path, output_path):
        """
        Initialize Flow Generator using TShark
        
        Args:
            pcap_path: Path to PCAP file
            output_path: Path to save flows CSV
        """
        self.pcap_path = pcap_path
        self.output_path = output_path
        self.flows = defaultdict(lambda: {
            'packets': 0,
            'bytes': 0,
            'start_time': None,
            'last_seen': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None
        })
    
    def extract_packets_with_tshark(self):
        """Extract packet data using tshark"""
        print(f"Reading PCAP with tshark: {self.pcap_path}")
        
        # Check if file exists
        if not os.path.exists(self.pcap_path):
            print(f"ERROR: PCAP file not found at {self.pcap_path}")
            return None
        
        # Check if file is empty
        if os.path.getsize(self.pcap_path) == 0:
            print(f"ERROR: PCAP file is empty")
            return None
        
        try:
            # Build tshark command to extract required fields
            cmd = [
                'tshark', '-r', self.pcap_path,
                '-T', 'fields',
                '-e', 'ip.src',
                '-e', 'ip.dst',
                '-e', 'tcp.srcport',
                '-e', 'tcp.dstport',
                '-e', 'udp.srcport',
                '-e', 'udp.dstport',
                '-e', 'ip.proto',
                '-e', 'frame.len',
                '-e', 'frame.time_epoch',
                '-E', 'separator=,'
            ]
            
            # Run tshark command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"ERROR: tshark failed with error: {result.stderr}")
                return None
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            if not lines or (len(lines) == 1 and lines[0] == ''):
                print(f"WARNING: No packets found in PCAP")
                return None
            
            print(f"Extracted {len(lines)} packets from PCAP")
            return lines
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: tshark timed out (file may be too large)")
            return None
        except Exception as e:
            print(f"ERROR: Failed to extract packets: {e}")
            return None
    
    def parse_packet_line(self, line):
        """Parse a single line of tshark output into packet fields"""
        parts = line.split(',')
        
        # Handle incomplete lines
        if len(parts) < 9:
            return None
        
        # Extract fields
        src_ip = parts[0] if parts[0] else None
        dst_ip = parts[1] if parts[1] else None
        tcp_src = parts[2] if parts[2] else None
        tcp_dst = parts[3] if parts[3] else None
        udp_src = parts[4] if parts[4] else None
        udp_dst = parts[5] if parts[5] else None
        protocol_num = parts[6] if parts[6] else None
        length = parts[7] if parts[7] else '0'
        timestamp = parts[8] if parts[8] else None
        
        # Skip packets without IP addresses
        if not src_ip or not dst_ip:
            return None
        
        # Skip IPv6 for simplicity
        if ':' in src_ip:
            return None
        
        # Determine protocol and ports
        if tcp_src and tcp_src != '':
            protocol = 'TCP'
            src_port = tcp_src
            dst_port = tcp_dst
        elif udp_src and udp_src != '':
            protocol = 'UDP'
            src_port = udp_src
            dst_port = udp_dst
        else:
            # Skip non-TCP/UDP packets
            return None
        
        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'length': int(length) if length else 0,
            'timestamp': float(timestamp) if timestamp else None
        }
    
    def process_packets(self):
        """Process all packets and build flows"""
        print(f"Processing PCAP file: {self.pcap_path}")
        
        # Extract packets using tshark
        packet_lines = self.extract_packets_with_tshark()
        
        if packet_lines is None:
            return
        
        packet_count = 0
        processed_packets = 0
        
        for line in packet_lines:
            if not line.strip():
                continue
            
            packet_count += 1
            
            # Parse the packet line
            packet = self.parse_packet_line(line)
            
            if packet is None:
                continue
            
            processed_packets += 1
            
            if processed_packets % 1000 == 0:
                print(f"Processed {processed_packets} packets...")
            
            # Generate flow key (5-tuple)
            flow_key = (
                packet['src_ip'], packet['dst_ip'],
                packet['src_port'], packet['dst_port'],
                packet['protocol']
            )
            
            # Update flow information
            flow = self.flows[flow_key]
            
            if flow['start_time'] is None:
                flow['start_time'] = packet['timestamp']
                flow['src_ip'] = packet['src_ip']
                flow['dst_ip'] = packet['dst_ip']
                flow['src_port'] = packet['src_port']
                flow['dst_port'] = packet['dst_port']
                flow['protocol'] = packet['protocol']
            
            flow['last_seen'] = packet['timestamp']
            flow['packets'] += 1
            flow['bytes'] += packet['length']
        
        print(f"Total packets processed: {processed_packets}")
        print(f"Total flows generated: {len(self.flows)}")
    
    def compute_flow_features(self):
        """Compute flow-level features"""
        flow_features = []
        
        for key, flow in self.flows.items():
            if flow['start_time'] is None or flow['last_seen'] is None:
                continue
            
            # Calculate duration
            duration = flow['last_seen'] - flow['start_time']
            if duration <= 0:
                duration = 0.001  # Minimum duration
            
            # Calculate packet rate (packets per second)
            packet_rate = flow['packets'] / duration
            
            # Calculate byte rate (bytes per second)
            byte_rate = flow['bytes'] / duration
            
            # Calculate average packet size
            avg_packet_size = flow['bytes'] / flow['packets'] if flow['packets'] > 0 else 0
            
            # Create feature dictionary with all required fields
            features = {
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'src_port': flow['src_port'],
                'dst_port': flow['dst_port'],
                'protocol': flow['protocol'],
                'packets': flow['packets'],
                'bytes': flow['bytes'],
                'duration': duration,
                'packet_rate': packet_rate,
                'byte_rate': byte_rate,
                'avg_packet_size': avg_packet_size
            }
            
            flow_features.append(features)
        
        return flow_features
    
    def save_flows(self):
        """Save flows to CSV file with pcap_source and capture_time"""
        flow_features = self.compute_flow_features()
        
        if not flow_features:
            print("WARNING: No flows to save")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(flow_features)
        
        # Add pcap_source and capture_time columns (like main project)
        df['pcap_source'] = os.path.basename(self.pcap_path)
        df['capture_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns to match main project structure
        column_order = [
            'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
            'packets', 'bytes', 'duration', 'packet_rate', 'byte_rate', 'avg_packet_size',
            'pcap_source', 'capture_time'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = 0 if col not in ['src_ip', 'dst_ip', 'protocol', 'pcap_source', 'capture_time'] else ''
        
        df = df[column_order]
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Check if file exists to append (like main project)
        if os.path.exists(self.output_path):
            try:
                existing = pd.read_csv(self.output_path)
                # Remove old entries from same PCAP if any
                if 'pcap_source' in existing.columns:
                    existing = existing[existing['pcap_source'] != os.path.basename(self.pcap_path)]
                df = pd.concat([existing, df], ignore_index=True)
            except:
                pass
        
        df.to_csv(self.output_path, index=False)
        print(f"Flows saved to: {self.output_path}")
        print(f"Saved {len(df)} flows (added {len(flow_features)} new)")
        
        return df
    
    def run(self):
        """Execute the complete flow generation pipeline"""
        self.process_packets()
        df = self.save_flows()
        return df


def main():
    """Test the flow generator"""
    print("="*60)
    print("TShark Flow Generator - Test")
    print("="*60)
    
    # Default paths
    pcap_path = 'data/raw_pcaps/sample.pcap'
    output_path = 'data/processed_flows/flows.csv'
    
    # Check if PCAP exists
    if not os.path.exists(pcap_path):
        print(f"ERROR: PCAP file not found at {pcap_path}")
        print("\nPlease add a PCAP file to data/raw_pcaps/")
        print("Or capture one using: tshark -i Wi-Fi -a duration:30 -w data/raw_pcaps/sample.pcap")
        return
    
    # Create generator and run
    generator = FlowGenerator(pcap_path, output_path)
    df = generator.run()
    
    if df is not None:
        print("\n" + "="*60)
        print("Flow Summary")
        print("="*60)
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show flow statistics
        print("\nFlow Statistics:")
        print(f"  Total flows: {len(df)}")
        print(f"  TCP flows: {len(df[df['protocol'] == 'TCP'])}")
        print(f"  UDP flows: {len(df[df['protocol'] == 'UDP'])}")
        print(f"  Total packets: {df['packets'].sum()}")
        print(f"  Total bytes: {df['bytes'].sum():,}")
        print(f"  Avg packet rate: {df['packet_rate'].mean():.2f} pps")
        print(f"  Avg byte rate: {df['byte_rate'].mean():.2f} Bps")


if __name__ == "__main__":
    main()