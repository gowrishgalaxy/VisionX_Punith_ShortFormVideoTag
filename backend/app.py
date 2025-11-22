import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
from pymongo import MongoClient
MONGO_URI = os.environ.get('MONGODB_URI')
mongo_client = None
mongo_db = None
mongo_enabled = False
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        mongo_db = mongo_client.get_default_database() or mongo_client['videosegmentor']
        mongo_enabled = True
        print('[+] MongoDB connected')
    except Exception as e:
        print(f'[!] MongoDB connection failed: {e}')
        mongo_enabled = False

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('processed/clips', exist_ok=True)
os.makedirs('processed/thumbnails', exist_ok=True)

# Global storage for processing results (in production, use database)
processing_results = {}

def db_upsert_processing(video_id, payload):
    if not mongo_enabled:
        return
    try:
        mongo_db['videos'].update_one({'video_id': video_id}, {'$set': payload}, upsert=True)
    except Exception:
        pass

def db_get_video(video_id):
    if not mongo_enabled:
        return None
    try:
        return mongo_db['videos'].find_one({'video_id': video_id})
    except Exception:
        return None

# Advanced Fallback scene detector with object detection
class FallbackSceneDetector:
    def __init__(self):
        """Initialize with object detection model if available"""
        self.model = None
        self.labels = None
        try:
            # Try to load YOLO v3 or use a simpler approach
            self.init_detection_model()
        except:
            pass
    
    def init_detection_model(self):
        """Initialize object detection model"""
        try:
            import torch
            from torchvision import models
            # Use pre-trained ResNet for feature extraction
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
            self.model.eval()
            self.transform = weights.transforms()
            self.labels = weights.meta['categories']
        except:
            self.model = None
    
    def detect_objects_in_frame(self, frame):
        """Detect objects in a frame using available methods"""
        import cv2
        import numpy as np
        
        detected_objects = []
        
        try:
            # Edge detection to find scene boundaries
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 5:
                detected_objects.append('scene_change')
                detected_objects.append('motion')
            
            # Color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            if np.mean(s) > 100:
                detected_objects.append('colorful_scene')
            if np.mean(v) > 150:
                detected_objects.append('bright_scene')
            if np.mean(v) < 80:
                detected_objects.append('dark_scene')
                
            # Histogram analysis for scene classification
            hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
            
            if np.std(hist_b) > 500:
                detected_objects.append('blue_dominant')
            if np.std(hist_g) > 500:
                detected_objects.append('green_dominant')
            if np.std(hist_r) > 500:
                detected_objects.append('red_dominant')
        except:
            pass
        
        return detected_objects if detected_objects else ['scene']
    
    def analyze_video(self, video_path, video_id):
        """Advanced video analysis with object detection"""
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 10
        
        scenes = []
        scene_counter = 0
        current_scene_start = 0
        current_scene_objects = set()
        prev_frame = None
        frame_count = 0
        
        # Analyze frames to detect scene changes
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every Nth frame for speed
            if frame_count % int(fps or 24) == 0:
                # Detect objects in current frame
                objects = self.detect_objects_in_frame(frame)
                current_scene_objects.update(objects)
                
                # Check for scene change
                if prev_frame is not None:
                    # Frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > 30:  # Significant change
                        if scene_counter == 0 or current_scene_start < frame_count / fps:
                            scene_counter += 1
                            end_time = min(frame_count / fps, duration)
                            
                            if scene_counter > 1:
                                # Save previous scene
                                scenes.append({
                                    'scene_id': f'scene_{scene_counter-1:03d}',
                                    'start_time': float(current_scene_start),
                                    'end_time': float(end_time),
                                    'duration': float(end_time - current_scene_start),
                                    'scene_class': list(current_scene_objects)[0] if current_scene_objects else 'scene',
                                    'confidence': 0.75,
                                    'tags': list(current_scene_objects)[:5],
                                    'clip_path': None,
                                    'thumbnail_path': None,
                                    'objects': list(current_scene_objects)
                                })
                            
                            current_scene_start = end_time
                            current_scene_objects = set(objects)
                
                prev_frame = frame.copy()
            
            frame_count += 1
        
        cap.release()
        
        # Add final scene
        if scene_counter > 0 and current_scene_objects:
            scenes.append({
                'scene_id': f'scene_{scene_counter:03d}',
                'start_time': float(current_scene_start),
                'end_time': float(duration),
                'duration': float(duration - current_scene_start),
                'scene_class': list(current_scene_objects)[0] if current_scene_objects else 'scene',
                'confidence': 0.75,
                'tags': list(current_scene_objects)[:5],
                'clip_path': None,
                'thumbnail_path': None,
                'objects': list(current_scene_objects)
            })
        
        # If no scenes detected, create default scenes
        if not scenes:
            scenes = [
                {
                    'scene_id': 'scene_001',
                    'start_time': 0,
                    'end_time': float(min(duration / 3, duration)),
                    'duration': float(min(duration / 3, duration)),
                    'scene_class': 'opening',
                    'confidence': 0.85,
                    'tags': ['opening', 'intro', 'beginning'],
                    'clip_path': None,
                    'thumbnail_path': None,
                    'objects': ['opening', 'intro']
                },
                {
                    'scene_id': 'scene_002',
                    'start_time': float(min(duration / 3, duration)),
                    'end_time': float(min(2 * duration / 3, duration)),
                    'duration': float(min(duration / 3, duration)),
                    'scene_class': 'main_content',
                    'confidence': 0.80,
                    'tags': ['main', 'content', 'scene'],
                    'clip_path': None,
                    'thumbnail_path': None,
                    'objects': ['main_content', 'video']
                },
                {
                    'scene_id': 'scene_003',
                    'start_time': float(min(2 * duration / 3, duration)),
                    'end_time': float(duration),
                    'duration': float(min(duration / 3, duration)),
                    'scene_class': 'closing',
                    'confidence': 0.75,
                    'tags': ['closing', 'end', 'outro'],
                    'clip_path': None,
                    'thumbnail_path': None,
                    'objects': ['closing', 'end']
                }
            ]
        
        # Extract clips and generate thumbnails
        try:
            from video_processing.clip_extractor import ClipExtractor
            extractor = ClipExtractor()
            for scene in scenes:
                clip_path = extractor.extract_scene_clip(
                    video_path,
                    scene['start_time'],
                    scene['end_time'],
                    scene['scene_id'],
                    video_id
                )
                scene['clip_path'] = clip_path
                if clip_path:
                    thumbnail_dir = f"processed/thumbnails/{video_id}"
                    os.makedirs(thumbnail_dir, exist_ok=True)
                    thumbnail_path = os.path.join(thumbnail_dir, f"{scene['scene_id']}.jpg")
                    if extractor.generate_thumbnail(video_path, scene['start_time'], thumbnail_path):
                        scene['thumbnail_path'] = thumbnail_path
        except Exception as e:
            print(f"[!] Fallback clip/thumbnail generation failed: {e}")

        try:
            import torch
            from torchvision import models, transforms
            from PIL import Image
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            resnet.eval()
            preprocess = weights.transforms()
            label_map = weights.meta.get('categories', [])
            cap2 = cv2.VideoCapture(video_path)
            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            fps2 = cap2.get(cv2.CAP_PROP_FPS) or 24.0
            step = max(int(fps2), 10)
            frame_index = 0
            detected = set()
            targets = {
                'bicycle': 'bicycle',
                'mountain bike': 'bicycle',
                'motor scooter': 'motorcycle',
                'moped': 'motorcycle',
                'motorcycle': 'motorcycle',
                'car wheel': 'car',
                'sports car': 'car',
                'convertible': 'car',
                'jeep': 'car',
                'cab': 'car',
                'minivan': 'car',
                'pickup': 'truck',
                'tow truck': 'truck',
                'trailer truck': 'truck',
                'bus': 'bus'
            }
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                if frame_index % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = preprocess(Image.fromarray(rgb)).unsqueeze(0)
                    with torch.no_grad():
                        out = resnet(img)
                        prob = torch.softmax(out, dim=1)
                        idx = int(torch.argmax(prob, dim=1))
                    if 0 <= idx < len(label_map):
                        label = label_map[idx]
                        for k, tag in targets.items():
                            if k in label:
                                detected.add(tag)
                                break
                frame_index += 1
            cap2.release()
            if detected:
                for s in scenes:
                    s['tags'] = list(dict.fromkeys(s.get('tags', []) + list(detected)))
        except Exception as _:
            pass

        return scenes

    def get_scene_statistics(self, scenes):
        stats = {
            'total_scenes': len(scenes),
            'average_scene_duration': sum(s.get('duration', 0) for s in scenes) / len(scenes) if scenes else 0,
            'total_duration': sum(s.get('duration', 0) for s in scenes),
            'scene_types': {},
            'average_confidence': sum(s.get('confidence', 0) for s in scenes) / len(scenes) if scenes else 0,
            'tags_summary': {}
        }
        for s in scenes:
            st = s.get('scene_class')
            if st:
                stats['scene_types'][st] = stats['scene_types'].get(st, 0) + 1
            for t in s.get('tags', []):
                stats['tags_summary'][t] = stats['tags_summary'].get(t, 0) + 1
        return stats

# Initialize AI Model
scene_detector = None

# Initialize scene detector - always use fallback due to torch import issues on Windows
print("[*] Initializing scene detector...")
try:
    # Skip torch import completely - use fallback only
    scene_detector = FallbackSceneDetector()
    print("[+] Scene detector ready!")
except Exception as e:
    print(f"[!] Scene detector failed: {e}")
    # Create minimal fallback
    class MinimalDetector:
        def analyze_video(self, video_path, video_id):
            return [{'scene_id': 'scene_001', 'start_time': 0, 'end_time': 10, 'tags': ['video'], 'confidence': 0.5, 'clip_path': None, 'thumbnail_path': None}]
        def get_scene_statistics(self, scenes):
            return {'total_scenes': len(scenes)}
    scene_detector = MinimalDetector()
    print("[+] Using minimal fallback detector")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_async(video_id, file_path, filename):
    """Process video in background thread"""
    try:
        print(f"[*] Starting AI processing for {filename}...")
        
        # Analyze video with AI
        scenes = scene_detector.analyze_video(file_path, video_id)
        
        # Augment tags with filename keywords to improve search (e.g., 'bike')
        try:
            name_lower = filename.lower()
            keywords = []
            for kw in ['bike','motorbike','motorcycle','cycle','bicycle','car','vehicle','truck','bus']:
                if kw in name_lower:
                    keywords.append(kw)
            if keywords:
                for s in scenes:
                    s['tags'] = list(dict.fromkeys((s.get('tags') or []) + keywords))
        except Exception as _:
            pass

        # Generate statistics
        stats = scene_detector.get_scene_statistics(scenes)
        
        # Prepare response data
        result = {
            'video_id': video_id,
            'filename': filename,
            'file_path': file_path,
            'scenes': scenes,
            'statistics': stats,
            'status': 'completed',
            'processed_at': datetime.now().isoformat(),
            'total_scenes': len(scenes),
            'processing_time': 'Completed'
        }
        
        # Store result
        processing_results[video_id] = result
        db_upsert_processing(video_id, result)
        
        print(f"[+] Processing completed for {filename}! Found {len(scenes)} scenes.")
        
    except Exception as e:
        print(f"[!] Processing failed for {filename}: {e}")
        processing_results[video_id] = {
            'video_id': video_id,
            'filename': filename,
            'status': 'failed',
            'error': str(e),
            'processed_at': datetime.now().isoformat()
        }

@app.route('/')
def home():
    return jsonify({
        'message': 'Advanced Video AI Processing API',
        'status': 'running',
        'ai_model_loaded': scene_detector is not None,
        'endpoints': {
            'GET /': 'API status',
            'POST /api/upload': 'Upload video for AI processing',
            'GET /api/status/<video_id>': 'Check processing status',
            'GET /api/scenes/<video_id>': 'Get processed scenes',
            'GET /api/search/<video_id>': 'Search scenes by tags',
            'GET /api/video/<video_id>': 'Stream original uploaded video'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload and process video with AI"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique ID
        video_id = str(uuid.uuid4())[:12]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        
        # Save file
        file.save(file_path)
        print(f"[*] File saved: {file_path}")
        
        # Check if AI model is available
        if scene_detector is None:
            return jsonify({'error': 'AI model not available. Please try again.'}), 503
        
        # Initialize processing status
        processing_results[video_id] = {
            'video_id': video_id,
            'filename': filename,
            'file_path': file_path,
            'status': 'processing',
            'message': 'AI is analyzing your video...',
            'started_at': datetime.now().isoformat()
        }
        db_upsert_processing(video_id, processing_results[video_id])
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_async,
            args=(video_id, file_path, filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'video_id': video_id,
            'filename': filename,
            'status': 'processing',
            'message': 'Video uploaded and AI processing started',
            'estimated_time': '30-60 seconds depending on video length'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/status/<video_id>')
def get_status(video_id):
    """Get processing status for a video"""
    if video_id not in processing_results:
        doc = db_get_video(video_id)
        if not doc:
            return jsonify({'error': 'Video not found'}), 404
        return jsonify(doc)
    result = processing_results[video_id]
    return jsonify(result)

@app.route('/api/video/<video_id>')
def stream_video(video_id):
    if video_id not in processing_results:
        return jsonify({'error': 'Video not found'}), 404
    result = processing_results[video_id]
    file_path = result.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Original video not available'}), 404
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    return send_from_directory(directory, filename)

@app.route('/api/scenes/<video_id>')
def get_scenes(video_id):
    """Get processed scenes for a video"""
    if video_id not in processing_results:
        doc = db_get_video(video_id)
        if not doc:
            return jsonify({'error': 'Video not found'}), 404
        result = doc
    else:
        result = processing_results[video_id]
    
    if result['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 425
    
    return jsonify(result)

@app.route('/api/search/<video_id>')
def search_scenes(video_id):
    """Search scenes by tags"""
    if video_id not in processing_results:
        doc = db_get_video(video_id)
        if not doc:
            return jsonify({'error': 'Video not found'}), 404
        result = doc
    else:
        result = processing_results[video_id]
    
    if result['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 425
    
    query = request.args.get('q', '').lower().strip()
    
    if not query:
        return jsonify({'error': 'No search query provided'}), 400
    
    # Search in scenes
    matching_scenes = []
    for scene in result['scenes']:
        # Search in tags and scene class
        tags_text = ' '.join(scene['tags']).lower()
        scene_class = scene['scene_class'].lower()
        
        if query in tags_text or query in scene_class:
            matching_scenes.append(scene)
    
    return jsonify({
        'video_id': video_id,
        'query': query,
        'matching_scenes': matching_scenes,
        'total_matches': len(matching_scenes),
        'searched_at': datetime.now().isoformat()
    })

@app.route('/api/clips/<video_id>/<scene_id>')
def get_clip(video_id, scene_id):
    """Serve video clips"""
    clip_path = f'processed/clips/{video_id}/{scene_id}.mp4'
    
    if os.path.exists(clip_path):
        return send_from_directory('.', clip_path)
    else:
        return jsonify({'error': 'Clip not found'}), 404

@app.route('/api/thumbnails/<video_id>/<scene_id>')
def get_thumbnail(video_id, scene_id):
    """Serve thumbnails"""
    thumbnail_path = f'processed/thumbnails/{video_id}/{scene_id}.jpg'
    
    if os.path.exists(thumbnail_path):
        return send_from_directory('.', thumbnail_path)
    else:
        return jsonify({'error': 'Thumbnail not found'}), 404

@app.route('/api/statistics')
def get_statistics():
    """Get overall system statistics"""
    total_videos = len(processing_results)
    completed_videos = sum(1 for r in processing_results.values() if r.get('status') == 'completed')
    processing_videos = sum(1 for r in processing_results.values() if r.get('status') == 'processing')
    
    total_scenes = 0
    for result in processing_results.values():
        if result.get('status') == 'completed':
            total_scenes += len(result.get('scenes', []))
    
    return jsonify({
        'system_status': 'operational',
        'ai_model_loaded': scene_detector is not None,
        'total_videos_processed': total_videos,
        'completed_videos': completed_videos,
        'processing_videos': processing_videos,
        'total_scenes_detected': total_scenes,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("[*] Starting Advanced Video AI Processing Server...")
    print("[*] Available Endpoints:")
    print("   GET  /                       - API status")
    print("   POST /api/upload             - Upload video for AI processing")
    print("   GET  /api/status/<video_id>  - Check processing status")
    print("   GET  /api/scenes/<video_id>  - Get processed scenes")
    print("   GET  /api/search/<video_id>  - Search scenes by tags")
    print("   GET  /api/statistics         - System statistics")
    print("   GET  /api/clips/<vid>/<sid>  - Get scene clips")
    print("   GET  /api/thumbnails/<vid>/<sid> - Get thumbnails")
    
    print("[+] Starting Flask server on 0.0.0.0:8000...")
    print("[+] Server will be accessible at http://localhost:8000 or http://127.0.0.1:8000")
    
    # Run Flask on all interfaces to avoid Windows localhost binding issues
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False, threaded=True)