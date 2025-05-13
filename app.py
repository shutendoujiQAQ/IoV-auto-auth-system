# app.py
# Flask application for monitoring Z3 solver input and output files

from flask import Flask, jsonify, render_template, send_from_directory
import json
import os
import time
import threading

app = Flask(__name__, static_folder='static', template_folder='templates')

# File paths
VLM_PATH = 'VLM.json'
DL_PATH = 'DL.json'
BUS_PATH = 'BUSDATA.json'
RESULT_PATH = 'result.json'

# Store last modification time
last_modified = {
    'vlm': 0,
    'dl': 0,
    'bus': 0,
    'result': 0
}

# Store file contents
file_contents = {
    'vlm': {},
    'dl': {},
    'bus': {},
    'result': {}
}

# Load JSON file
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

# Check if file has been updated
def is_file_updated(file_path, last_time):
    try:
        current_time = os.path.getmtime(file_path)
        return current_time > last_time, current_time
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False, last_time

# Update file contents
def update_file_contents():
    global last_modified, file_contents
    
    # Check VLM.json
    updated, new_time = is_file_updated(VLM_PATH, last_modified['vlm'])
    if updated:
        file_contents['vlm'] = load_json_file(VLM_PATH)
        last_modified['vlm'] = new_time
    elif not file_contents['vlm'] and os.path.exists(VLM_PATH):
        # Load file even if it hasn't changed, if content is empty
        file_contents['vlm'] = load_json_file(VLM_PATH)
    
    # Check DL.json
    updated, new_time = is_file_updated(DL_PATH, last_modified['dl'])
    if updated:
        file_contents['dl'] = load_json_file(DL_PATH)
        last_modified['dl'] = new_time
    elif not file_contents['dl'] and os.path.exists(DL_PATH):
        file_contents['dl'] = load_json_file(DL_PATH)
    
    # Check BUSDATA_PROCESSED.json
    updated, new_time = is_file_updated(BUS_PATH, last_modified['bus'])
    if updated:
        file_contents['bus'] = load_json_file(BUS_PATH)
        last_modified['bus'] = new_time
    elif not file_contents['bus'] and os.path.exists(BUS_PATH):
        file_contents['bus'] = load_json_file(BUS_PATH)
    
    # Check result.json
    updated, new_time = is_file_updated(RESULT_PATH, last_modified['result'])
    if updated:
        file_contents['result'] = load_json_file(RESULT_PATH)
        last_modified['result'] = new_time
    elif not file_contents['result'] and os.path.exists(RESULT_PATH):
        file_contents['result'] = load_json_file(RESULT_PATH)

# Initialize file contents
def init_file_contents():
    global file_contents, last_modified
    
    # Load VLM.json
    if os.path.exists(VLM_PATH):
        file_contents['vlm'] = load_json_file(VLM_PATH)
        last_modified['vlm'] = os.path.getmtime(VLM_PATH)
    
    # Load DL.json
    if os.path.exists(DL_PATH):
        file_contents['dl'] = load_json_file(DL_PATH)
        last_modified['dl'] = os.path.getmtime(DL_PATH)
    
    # Load BUSDATA_PROCESSED.json
    if os.path.exists(BUS_PATH):
        file_contents['bus'] = load_json_file(BUS_PATH)
        last_modified['bus'] = os.path.getmtime(BUS_PATH)
    
    # Load result.json
    if os.path.exists(RESULT_PATH):
        file_contents['result'] = load_json_file(RESULT_PATH)
        last_modified['result'] = os.path.getmtime(RESULT_PATH)

# Background thread, periodically check for file updates
def background_update():
    while True:
        update_file_contents()
        time.sleep(1)  # Check once per second

# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Get all data
@app.route('/api/data')
def get_data():
    return jsonify({
        'vlm': file_contents['vlm'],
        'dl': file_contents['dl'],
        'bus': file_contents['bus'],
        'result': file_contents['result']
    })

# Route: Get VLM data
@app.route('/api/vlm')
def get_vlm():
    return jsonify(file_contents['vlm'])

# Route: Get DL data
@app.route('/api/dl')
def get_dl():
    return jsonify(file_contents['dl'])

# Route: Get BUSDATA data
@app.route('/api/bus')
def get_bus():
    return jsonify(file_contents['bus'])

# Route: Get result data
@app.route('/api/result')
def get_result():
    return jsonify(file_contents['result'])

# Start application
if __name__ == '__main__':
    # Initialize file contents
    init_file_contents()
    
    # Start background thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Start Flask application
    app.run(debug=True, host='0.0.0.0', port=9000)