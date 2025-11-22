import os
import subprocess
import cv2

class ClipExtractor:
    def __init__(self, output_base_dir='processed/clips'):
        self.output_base_dir = output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
    
    def extract_scene_clip(self, video_path, start_time, end_time, scene_id, video_id):
        """Extract a clip for a specific scene"""
        try:
            # Create output directory for this video
            video_clips_dir = os.path.join(self.output_base_dir, video_id)
            os.makedirs(video_clips_dir, exist_ok=True)
            
            output_path = os.path.join(video_clips_dir, f"{scene_id}.mp4")
            
            # Calculate duration
            duration = end_time - start_time
            
            # Use ffmpeg to extract clip
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(output_path):
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    if frame_count and frame_count > 0:
                        return output_path
                    else:
                        try:
                            os.remove(output_path)
                        except:
                            pass
                # Fall through to OpenCV fallback
            
            # Fallback: use OpenCV to write frames
            cap_in = cv2.VideoCapture(video_path)
            if not cap_in.isOpened():
                return None
            fps_in = cap_in.get(cv2.CAP_PROP_FPS) or 24.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps_in, (int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            start_frame = int(start_time * fps_in)
            end_frame = int(end_time * fps_in)
            current = 0
            while True:
                ret, frame = cap_in.read()
                if not ret:
                    break
                if current >= start_frame and current <= end_frame:
                    out.write(frame)
                if current > end_frame:
                    break
                current += 1
            cap_in.release()
            out.release()
            if os.path.exists(output_path):
                cap_out = cv2.VideoCapture(output_path)
                ok = cap_out.isOpened() and cap_out.get(cv2.CAP_PROP_FRAME_COUNT) > 0
                cap_out.release()
                if ok:
                    return output_path
            return None
        
        except Exception as e:
            print(f"❌ Clip extraction failed for {scene_id}: {e}")
            return None
    
    def generate_thumbnail(self, video_path, timestamp, output_path, width=320):
        """Generate thumbnail for a scene"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(timestamp),
                '-vframes', '1',
                '-vf', f'scale={width}:-1',
                '-y',  # Overwrite output file
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(output_path):
                return True
            # Fallback using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            target_frame = int(timestamp * fps)
            current = 0
            frame_ok = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current >= target_frame:
                    frame_ok = True
                    break
                current += 1
            cap.release()
            if frame_ok:
                try:
                    import numpy as np
                    h, w = frame.shape[:2]
                    new_w = width
                    new_h = int(h * (new_w / w))
                    resized = cv2.resize(frame, (new_w, new_h))
                    cv2.imwrite(output_path, resized)
                    return os.path.exists(output_path)
                except Exception as _:
                    return False
            return False
        except Exception as e:
            print(f"❌ Thumbnail generation failed: {e}")
            return False