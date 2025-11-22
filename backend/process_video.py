#!/usr/bin/env python3
"""
Video processing worker script - called by Node.js Express server
Analyzes video and returns results as JSON
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

def analyze_video(video_path):
    """Analyze video and extract scenes with tags"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'error': 'Failed to open video file',
                'scenes': [],
                'tags': []
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample every N frames to speed up processing
        sample_rate = max(1, int(fps * 2))  # Sample every 2 seconds
        
        scenes = []
        all_tags = set()
        prev_frame = None
        frame_idx = 0
        scene_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process sampled frames
            if frame_idx % sample_rate == 0:
                # Detect scene changes
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow / motion
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    
                    # Scene change threshold
                    if motion > 30:
                        if scene_frames:
                            # Create scene object
                            start_time = (scene_frames[0] / fps)
                            end_time = (frame_idx / fps)
                            
                            scene_tags = list(all_tags) if all_tags else ['scene']
                            scenes.append({
                                'id': f'scene_{len(scenes)}',
                                'start_time': round(start_time, 2),
                                'end_time': round(end_time, 2),
                                'start_frame': scene_frames[0],
                                'end_frame': frame_idx,
                                'tags': scene_tags,
                                'confidence': 0.85
                            })
                            scene_frames = []
                            all_tags = set()
                
                # Extract color-based tags
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Detect blue (sky)
                lower_blue = np.array([100, 50, 50])
                upper_blue = np.array([130, 255, 255])
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
                if np.count_nonzero(mask_blue) > (frame.shape[0] * frame.shape[1] * 0.1):
                    all_tags.add('sky')
                
                # Detect green (nature)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([90, 255, 255])
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                if np.count_nonzero(mask_green) > (frame.shape[0] * frame.shape[1] * 0.15):
                    all_tags.add('nature')
                
                # Detect red/orange (fire/sunset)
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
                if np.count_nonzero(mask_red) > (frame.shape[0] * frame.shape[1] * 0.15):
                    all_tags.add('warm_tones')
                
                # Add brightness-based tags
                brightness = np.mean(gray)
                if brightness > 200:
                    all_tags.add('bright')
                elif brightness < 50:
                    all_tags.add('dark')
                
                scene_frames.append(frame_idx)
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        
        # Handle remaining scene
        if scene_frames:
            start_time = (scene_frames[0] / fps)
            end_time = (frame_idx / fps)
            scene_tags = list(all_tags) if all_tags else ['scene']
            scenes.append({
                'id': f'scene_{len(scenes)}',
                'start_time': round(start_time, 2),
                'end_time': round(end_time, 2),
                'start_frame': scene_frames[0],
                'end_frame': frame_idx,
                'tags': scene_tags,
                'confidence': 0.85
            })
        
        # If no scenes detected, create one for entire video
        if not scenes:
            duration = frame_idx / fps
            scenes.append({
                'id': 'scene_0',
                'start_time': 0,
                'end_time': round(duration, 2),
                'start_frame': 0,
                'end_frame': frame_idx,
                'tags': ['video'],
                'confidence': 0.9
            })
        
        # Extract unique tags
        all_unique_tags = set()
        for scene in scenes:
            all_unique_tags.update(scene['tags'])
        
        return {
            'success': True,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': round(total_frames / fps, 2)
            },
            'scenes': scenes,
            'tags': list(all_unique_tags),
            'total_scenes': len(scenes)
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'scenes': [],
            'tags': []
        }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python process_video.py <video_path> [video_id]'}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_id = sys.argv[2] if len(sys.argv) > 2 else 'unknown'
    
    if not Path(video_path).exists():
        print(json.dumps({'error': f'Video file not found: {video_path}'}))
        sys.exit(1)
    
    result = analyze_video(video_path)
    print(json.dumps(result))
    sys.exit(0)
