#!/usr/bin/env python
from flask import Flask, jsonify
import socket
import sys
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'status': 'ok', 'message': 'Test server working'})

if __name__ == '__main__':
    print("[+] Starting test Flask server...")
    sys.stdout.flush()
    
    # Disable IPv6
    socket.has_ipv6 = False
    
    try:
        print("[+] Binding to 127.0.0.1:8001...")
        app.run(host='127.0.0.1', port=8001, debug=False, use_reloader=False, use_debugger=False)
    except Exception as e:
        print(f"[!] Error: {e}")
        print("[+] Trying 0.0.0.0:8001...")
        app.run(host='0.0.0.0', port=8001, debug=False, use_reloader=False, use_debugger=False)

