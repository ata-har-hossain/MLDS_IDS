# fully_automated.py - Main Entry Point for Multi-Model IDS (NO EMOJIS)
import os
import sys
import subprocess
import webbrowser
import time
import ctypes
import threading

def is_admin():
    """Check if running as administrator"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_tshark():
    """Check if tshark is installed"""
    try:
        subprocess.run(['tshark', '--version'], capture_output=True, check=True)
        print("[OK] tshark found")
        return True
    except:
        print("[ERROR] tshark not found! Install Wireshark")
        print("   Download: https://www.wireshark.org/download.html")
        return False

def check_models():
    """Check if all model files exist"""
    model_dir = 'model'
    # Only check working models
    required_models = [
        'rf_model.pkl', 'xgb_model.json',
        'et_model.pkl', 'dt_model.pkl', 
        'gb_model.joblib', 'label_mapping.json'
    ]
    
    print("\n[CHECK] Verifying model files...")
    found = 0
    for model in required_models:
        if os.path.exists(f'{model_dir}/{model}'):
            print(f"   [OK] {model}")
            found += 1
        else:
            print(f"   [MISSING] {model}")
    
    print(f"\n   Models available: {found}/{len(required_models)}")
    return found

def main():
    print("="*60)
    print("   MACHINE LEARNING DRIVEN SMART INTRUSION DETECTION SYSTEM")
    print("   5 ML Models + Rule-Based Detection")
    print("="*60)
    print()
    
    # Check tshark
    if not check_tshark():
        sys.exit(1)
    
    # Check models
    models_found = check_models()
    
    # Check admin privileges
    if not is_admin():
        print("\n[WARNING] Administrator privileges needed for live capture.")
        print("   The system will request admin rights when capturing.\n")
    
    print("\n" + "="*60)
    print("SELECT MODE")
    print("="*60)
    print("  [1] Process existing PCAPs (one-time)")
    print("  [2] Full system: Continuous capture + Auto-process + Dashboard")
    print("  [3] Launch Dashboard only")
    print("  [4] Process PCAPs and exit (no dashboard)")
    print()
    
    choice = input("Enter choice (1-4): ")
    
    if choice == '1':
        print("\n" + "="*60)
        print("Processing existing PCAPs...")
        print("="*60)
        subprocess.run(['python', 'auto_processor.py', '--once'])
        
    elif choice == '2':
        print("\n" + "="*60)
        print("Starting FULL SYSTEM...")
        print("="*60)
        print("\nComponents:")
        print("   [CAPTURE] Continuous Capture (30s intervals)")
        print("   [PROCESSOR] Auto Processor (5 ML Models)")
        print("   [DASHBOARD] Streamlit Dashboard")
        print("\nPress Ctrl+C to stop everything\n")
        
        # Start continuous capture
        def run_capture():
            try:
                subprocess.run(['python', 'continuous_capture.py'])
            except:
                pass
        
        # Start auto processor
        def run_processor():
            try:
                subprocess.run(['python', 'auto_processor.py'])
            except:
                pass
        
        # Start dashboard
        def run_dashboard():
            time.sleep(3)  # Wait for processor to initialize
            webbrowser.open('http://localhost:8501')
            subprocess.run(['streamlit', 'run', 'src/gui_auto.py', '--server.headless', 'true'])
        
        # Start all threads
        t1 = threading.Thread(target=run_capture, daemon=True)
        t2 = threading.Thread(target=run_processor, daemon=True)
        t3 = threading.Thread(target=run_dashboard, daemon=True)
        
        t1.start()
        t2.start()
        t3.start()
        
        print("\n[OK] SYSTEM RUNNING")
        print("[DASHBOARD] http://localhost:8501")
        print("\n" + "-"*40)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n[STOP] Stopping system...")
            print("[OK] Done.")
            
    elif choice == '3':
        print("\n" + "="*60)
        print("Launching Dashboard...")
        print("="*60)
        print("\n[DASHBOARD] http://localhost:8501")
        print("Press Ctrl+C to stop dashboard\n")
        
        webbrowser.open('http://localhost:8501')
        subprocess.run(['streamlit', 'run', 'src/gui_auto.py'])
        
    elif choice == '4':
        print("\n" + "="*60)
        print("Processing PCAPs (no dashboard)...")
        print("="*60)
        subprocess.run(['python', 'auto_processor.py'])
        
    else:
        print("[ERROR] Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] System stopped by user")
        sys.exit(0)