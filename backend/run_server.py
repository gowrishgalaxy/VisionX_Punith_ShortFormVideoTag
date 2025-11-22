#!/usr/bin/env python
import sys
import os

# Change to the backend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import Flask app
from app import app

if __name__ == '__main__':
    print("[+] Starting Flask server...")
    sys.stdout.flush()
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
