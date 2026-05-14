# controller.py - Updated for 5 ML Models (Random Forest, XGBoost, Extra Trees, Decision Tree, Gradient Boosting)
import subprocess
import os
import signal
import time
import sys
import threading
import webbrowser

class ProcessController:
    def __init__(self):
        self.processes = []
        self.running = True
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n\nSTOPPING ALL PROCESSES...")
        self.running = False
        self.stop_all()
        sys.exit(0)
    
    def start_process(self, command, name):
        print(f"   Starting {name}...")
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            self.processes.append({'proc': proc, 'name': name, 'command': command})
            return proc
        except Exception as e:
            print(f"   [ERROR] Failed to start {name}: {e}")
            return None
    
    def stop_all(self):
        print("\n   Stopping processes...")
        
        for p in self.processes:
            proc = p['proc']
            name = p['name']
            print(f"      Terminating {name}...")
            try:
                proc.terminate()
            except:
                pass
        
        time.sleep(2)
        
        for p in self.processes:
            proc = p['proc']
            name = p['name']
            if proc.poll() is None:
                print(f"      Force killing {name}...")
                try:
                    proc.kill()
                except:
                    pass
        
        if sys.platform == 'win32':
            print("      Cleaning up orphaned processes...")
            subprocess.run(['taskkill', '/f', '/im', 'tshark.exe'], capture_output=True)
            subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], capture_output=True)
        
        print("   All processes stopped")
    
    def monitor_output(self):
        def read_output(proc, name):
            for line in iter(proc.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    if line:
                        print(f"[{name}] {line}")
                if not self.running:
                    break
        
        threads = []
        for p in self.processes:
            proc = p['proc']
            name = p['name']
            if proc and proc.stdout:
                t = threading.Thread(target=read_output, args=(proc, name))
                t.daemon = True
                t.start()
                threads.append(t)
        
        try:
            while self.running:
                time.sleep(1)
                
                for p in self.processes:
                    proc = p['proc']
                    name = p['name']
                    if proc and proc.poll() is not None:
                        print(f"[!] {name} has stopped (exit code: {proc.returncode})")
                        
                        if name in ['Processor', 'Dashboard'] and self.running:
                            print(f"[!] Auto-restarting {name}...")
                            new_proc = self.start_process(p['command'], name)
                            if new_proc:
                                p['proc'] = new_proc
                                t = threading.Thread(target=read_output, args=(new_proc, name))
                                t.daemon = True
                                t.start()
                                threads.append(t)
                
        except KeyboardInterrupt:
            self.signal_handler(None, None)
    
    def check_models(self):
        model_dir = 'model'
        # Only check working models
        required_models = ['rf_model.pkl', 'xgb_model.json', 
                          'et_model.pkl', 'dt_model.pkl', 
                          'gb_model.joblib', 'label_mapping.json']
        
        print("\nChecking model files...")
        found = 0
        for model in required_models:
            if os.path.exists(f'{model_dir}/{model}'):
                print(f"   [OK] {model}")
                found += 1
            else:
                print(f"   [MISSING] {model}")
        
        print(f"\n   Models available: {found}/{len(required_models)}")
        return found
    
    def run(self):
        project_dir = r"C:\Users\ataha\IoT_Project"
        if os.path.exists(project_dir):
            os.chdir(project_dir)
            print(f"Working directory: {project_dir}")
        else:
            print(f"Warning: Directory not found: {project_dir}")
        
        print("\n" + "="*60)
        print("MACHINE LEARNING DRIVEN SMART INTRUSION DETECTION SYSTEM")
        print("5 ML Models + Rule-Based Detection")
        print("="*60)
        
        models_found = self.check_models()
        
        print("\n" + "="*60)
        print("STARTING SERVICES")
        print("="*60)
        
        os.makedirs('data/raw_pcaps', exist_ok=True)
        os.makedirs('data/processed_flows', exist_ok=True)
        os.makedirs('model', exist_ok=True)
        
        self.start_process(['python', 'continuous_capture.py'], 'Capture')
        time.sleep(2)
        
        self.start_process(['python', 'auto_processor.py'], 'Processor')
        time.sleep(3)
        
        def open_dashboard():
            time.sleep(5)
            webbrowser.open('http://localhost:8501')
        
        threading.Thread(target=open_dashboard, daemon=True).start()
        self.start_process(['streamlit', 'run', 'src/gui_auto.py', '--server.headless', 'true', '--server.port', '8501'], 'Dashboard')
        
        print("\n" + "="*60)
        print("SYSTEM RUNNING")
        print("="*60)
        print(f"Capture: Active (30s intervals)")
        print(f"Processor: Active ({models_found} ML models loaded)")
        print(f"Dashboard: http://localhost:8501")
        print("\nAvailable ML Models:")
        if os.path.exists('model/rf_model.pkl'): print("   Random Forest")
        if os.path.exists('model/xgb_model.json'): print("   XGBoost")
        if os.path.exists('model/et_model.pkl'): print("   Extra Trees")
        if os.path.exists('model/dt_model.pkl'): print("   Decision Tree")
        if os.path.exists('model/gb_model.joblib'): print("   Gradient Boosting")
        print("\n" + "-"*40)
        print("Press Ctrl+C to stop everything")
        print("-"*40)
        
        self.monitor_output()


if __name__ == "__main__":
    controller = ProcessController()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.signal_handler(None, None)