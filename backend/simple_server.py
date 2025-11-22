#!/usr/bin/env python
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import os

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/':
            response = {'status': 'ok', 'message': 'Backend is working!', 'api_version': '1.0'}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
    
    def log_message(self, format, *args):
        print(format % args)

if __name__ == '__main__':
    server_address = ('127.0.0.1', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print("[+] Backend server running on http://127.0.0.1:8000")
    print("[+] Press Ctrl+C to stop")
    httpd.serve_forever()
