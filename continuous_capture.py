# continuous_capture.py - Continuous Capture (30 seconds)
import subprocess
import time
import os
from datetime import datetime
import ctypes
import sys

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

class ContinuousCapture:
    def __init__(self, output_dir='data/raw_pcaps', duration=30, interface='Wi-Fi'):
        self.output_dir = output_dir
        self.duration = duration
        self.interface = interface
        self.running = True
        os.makedirs(output_dir, exist_ok=True)
    
    def capture(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.output_dir}/capture_{timestamp}.pcap'
        
        print(f"\n📡 Capturing {self.duration}s on {self.interface} -> {os.path.basename(filename)}")
        
        cmd = ['tshark', '-i', self.interface, '-a', f'duration:{self.duration}', '-w', filename]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            size = os.path.getsize(filename) / 1024
            print(f"   ✅ Saved {size:.1f} KB")
            return filename
        else:
            if os.path.exists(filename):
                os.remove(filename)
            print(f"   ⚠️ No traffic")
            return None
    
    def run(self):
        print("="*60)
        print("CONTINUOUS CAPTURE")
        print("="*60)
        print(f"Interface: {self.interface}")
        print(f"Duration: {self.duration} seconds per file")
        print(f"Output: {self.output_dir}")
        print("="*60)
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while self.running:
                self.capture()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', default='Wi-Fi')
    parser.add_argument('-d', '--duration', type=int, default=30, help='Capture duration in seconds')
    parser.add_argument('-o', '--output', default='data/raw_pcaps')
    
    args = parser.parse_args()
    
    if not is_admin():
        print("Requesting administrator privileges for capture...")
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, " ".join(sys.argv), None, 1
        )
        return
    
    capturer = ContinuousCapture(args.output, args.duration, args.interface)
    capturer.run()


if __name__ == "__main__":
    main()