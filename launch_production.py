#!/usr/bin/env python3
"""
Production System Launcher
Starts Django server and ML production system
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

def start_django_server():
    """Start Django development server"""
    try:
        print("🌐 Starting Django server...")
        subprocess.run([
            sys.executable, "manage.py", "runserver", "127.0.0.1:8000"
        ], cwd=Path.cwd())
    except Exception as e:
        print(f"Django server error: {e}")

def start_production_system():
    """Start the ML production system"""
    try:
        print("🤖 Starting Production ML System...")
        time.sleep(3)  # Wait for Django to start
        
        from production_system import ProductionSystem
        system = ProductionSystem()
        system.start_production_system()
        
    except Exception as e:
        print(f"Production system error: {e}")

def main():
    """Launch complete production system"""
    print("🇳🇵 NEPAL SOCIAL SERVICE AI - PRODUCTION LAUNCHER")
    print("="*60)
    print("LAUNCHING:")
    print("1. Django Web Server (Port 8000)")
    print("2. ML Production System (Live Learning)")
    print("3. Real-time Dashboard")
    print("="*60)
    
    # Start Django server in background thread
    django_thread = threading.Thread(target=start_django_server, daemon=True)
    django_thread.start()
    
    # Start production system in main thread
    production_thread = threading.Thread(target=start_production_system)
    production_thread.start()
    
    print("\n🚀 SYSTEM LAUNCHED!")
    print("📊 Dashboard: http://127.0.0.1:8000/dashboard/")
    print("📈 API Status: http://127.0.0.1:8000/api/status/")
    print("🔄 Live System: Running in background")
    print("\nPress Ctrl+C to stop all systems")
    
    try:
        production_thread.join()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all systems...")

if __name__ == "__main__":
    main()