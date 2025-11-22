import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import json
import time

class AdvancedSceneDetector:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Loading Advanced AI Model on: {self.device}")
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model architecture
        from custom_model import AdvancedSceneCNN
        arch = self.config.get('model_architecture', {})
        num_classes = arch.get('num_classes') or self.config.get('num_classes') or 6
        self.model = AdvancedSceneCNN(num_classes=num_classes)
        
        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Advanced model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = arch.get('class_names', self.config.get('class_names', []))
        self.tag_categories = self.config.get('tag_categories', {})
        
        print(f"📊 Model configured for {len(self.class_names)} scene types")
    
    def analyze_video(self, video_path, video_id):
        """Comprehensive video analysis with scene detection and tagging"""
        print(f"🎬 Analyzing video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"   📊 Video info: {duration:.2f}s, {fps:.1f} FPS, {total_frames} frames")
        
        scenes = []
        frame_count = 0
        current_scene = None
        prev_frame = None
        scene_counter = 0
        
        # Initialize clip extractor
        from clip_extractor import ClipExtractor
        clip_extractor = ClipExtractor()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for efficiency (adjust based on needs)
            if frame_count % 3 == 0:
                # Convert frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict scene type
                prediction = self._predict_frame(frame_rgb)
                scene_class = prediction['class']
                confidence = prediction['confidence']
                
                # Scene change detection
                scene_change = self._detect_scene_change(
                    current_scene, scene_class, confidence, frame_count, fps, prev_frame, frame
                )
                
                if scene_change:
                    # Save previous scene if it meets minimum duration
                    if current_scene is not None:
                        scene_duration = current_scene['end_time'] - current_scene['start_time']
                        if scene_duration >= 1.0:  # Minimum 1 second duration
                            scenes.append(current_scene)
                    
                    # Start new scene
                    scene_counter += 1
                    current_scene = {
                        'scene_id': f"scene_{scene_counter:03d}",
                        'start_frame': frame_count,
                        'end_frame': frame_count,
                        'start_time': frame_count / fps,
                        'end_time': frame_count / fps,
                        'scene_class': scene_class,
                        'confidence': confidence,
                        'tags': self.model.generate_scene_tags(scene_class, confidence, frame_rgb),
                        'dominant_color': self.model.get_dominant_color(frame_rgb),
                        'clip_path': None,
                        'thumbnail_path': None
                    }
                else:
                    # Continue current scene
                    current_scene['end_frame'] = frame_count
                    current_scene['end_time'] = frame_count / fps
                    # Update confidence with moving average
                    current_scene['confidence'] = (current_scene['confidence'] + confidence) / 2
                    # Update tags based on latest frame
                    current_scene['tags'] = self.model.generate_scene_tags(
                        scene_class, current_scene['confidence'], frame_rgb
                    )
            
            prev_frame = frame
            frame_count += 1
            
            # Progress update
            if frame_count % 150 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   🔄 Processing: {progress:.1f}% - Found {len(scenes)} scenes")
        
        # Add final scene if it meets criteria
        if current_scene is not None:
            scene_duration = current_scene['end_time'] - current_scene['start_time']
            if scene_duration >= 1.0:
                scenes.append(current_scene)
        
        cap.release()
        
        print(f"✅ Scene detection complete! Found {len(scenes)} scenes")
        
        # Extract clips and thumbnails for each scene
        print("🎥 Extracting scene clips and thumbnails...")
        for i, scene in enumerate(scenes):
            # Extract clip
            clip_path = clip_extractor.extract_scene_clip(
                video_path,
                scene['start_time'],
                scene['end_time'],
                scene['scene_id'],
                video_id
            )
            scene['clip_path'] = clip_path
            
            # Generate thumbnail
            if clip_path:
                thumbnail_dir = f"processed/thumbnails/{video_id}"
                os.makedirs(thumbnail_dir, exist_ok=True)
                thumbnail_path = os.path.join(thumbnail_dir, f"{scene['scene_id']}.jpg")
                
                if clip_extractor.generate_thumbnail(
                    video_path, scene['start_time'], thumbnail_path
                ):
                    scene['thumbnail_path'] = thumbnail_path
            
            print(f"   📹 {i+1}/{len(scenes)}: {scene['scene_id']} - {scene['scene_class']}")
        
        return scenes
    
    def _predict_frame(self, frame):
        """Predict scene type for a single frame"""
        try:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(frame_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            return {
                'class': self.class_names[predicted_idx.item()],
                'confidence': confidence.item(),
                'class_id': predicted_idx.item()
            }
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'class': 'transition', 'confidence': 0.5, 'class_id': 5}
    
    def _detect_scene_change(self, current_scene, new_class, confidence, frame_count, fps, prev_frame, current_frame):
        """Advanced scene change detection"""
        if current_scene is None:
            return True
        
        # Class change
        if new_class != current_scene['scene_class']:
            return True
        
        # Low confidence indicates uncertainty/transition
        if confidence < 0.4:
            return True
        
        # Time-based change (if no change for 10 seconds)
        current_time = frame_count / fps
        scene_start_time = current_scene['start_time']
        if current_time - scene_start_time > 10.0:
            return True
        
        # Visual change detection using frame difference
        if prev_frame is not None and current_frame is not None:
            try:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
                frame_diff = cv2.absdiff(prev_gray, curr_gray)
                diff_score = np.mean(frame_diff)
                
                # Significant visual change
                if diff_score > 30:
                    return True
            except:
                pass
        
        return False
    
    def get_scene_statistics(self, scenes):
        """Generate statistics about detected scenes"""
        stats = {
            'total_scenes': len(scenes),
            'total_duration': sum(scene['end_time'] - scene['start_time'] for scene in scenes),
            'scene_types': {},
            'average_confidence': 0,
            'tags_summary': {}
        }
        
        if scenes:
            stats['average_confidence'] = sum(scene['confidence'] for scene in scenes) / len(scenes)
            
            # Count scene types
            for scene in scenes:
                scene_type = scene['scene_class']
                if scene_type not in stats['scene_types']:
                    stats['scene_types'][scene_type] = 0
                stats['scene_types'][scene_type] += 1
                
                # Count tags
                for tag in scene['tags']:
                    if tag not in stats['tags_summary']:
                        stats['tags_summary'][tag] = 0
                    stats['tags_summary'][tag] += 1
        
        return stats