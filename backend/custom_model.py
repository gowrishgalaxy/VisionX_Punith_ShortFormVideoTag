import torch.nn as nn
import torch.nn.functional as F

class AdvancedSceneCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AdvancedSceneCNN, self).__init__()
        
        # Enhanced feature extraction with multiple branches
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.texture_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Main processing branch
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 128 * 14 * 14 + 64 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Color features
        color_features = self.color_branch(x)
        color_features = color_features.view(color_features.size(0), -1)
        
        # Texture features
        texture_features = self.texture_branch(x)
        texture_features = texture_features.view(texture_features.size(0), -1)
        
        # Main features
        main_features = self.main_branch(x)
        main_features = main_features.view(main_features.size(0), -1)
        
        # Feature fusion
        combined = torch.cat([main_features, color_features, texture_features], dim=1)
        output = self.fusion(combined)
        
        return output

    def generate_scene_tags(self, scene_class, confidence, frame):
        """Generate descriptive tags for the scene"""
        import numpy as np
        import cv2
        
        base_tags = {
            'animation': ['cartoon', 'animated', 'graphics', 'digital_art', 'drawing'],
            'outdoor': ['nature', 'outside', 'landscape', 'environment', 'sky', 'outdoors'],
            'indoor': ['inside', 'room', 'building', 'interior', 'walls', 'furniture'],
            'action': ['movement', 'dynamic', 'fast_paced', 'exciting', 'motion', 'speed'],
            'dialogue': ['talking', 'conversation', 'people', 'communication', 'speaking', 'discussion'],
            'transition': ['change', 'shift', 'fade', 'cut', 'switch', 'transition']
        }
        
        # Get base tags for the scene class
        tags = base_tags.get(scene_class, ['video', 'scene', 'content'])
        
        # Add confidence-based tags
        if confidence > 0.8:
            tags.extend(['high_confidence', 'clear_scene', 'well_defined'])
        elif confidence < 0.5:
            tags.extend(['low_confidence', 'uncertain', 'ambiguous'])
        else:
            tags.extend(['medium_confidence', 'reasonable'])
        
        # Add color-based tags
        if frame is not None:
            try:
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                avg_saturation = np.mean(frame_hsv[:,:,1])
                avg_brightness = np.mean(frame_hsv[:,:,2])
                
                if avg_brightness > 180:
                    tags.extend(['bright', 'well_lit', 'luminous'])
                elif avg_brightness < 80:
                    tags.extend(['dark', 'low_light', 'dim'])
                    
                if avg_saturation > 120:
                    tags.extend(['colorful', 'vibrant', 'saturated'])
                elif avg_saturation < 50:
                    tags.extend(['muted', 'desaturated', 'subtle'])
            except:
                pass
        
        # Add duration-based tags (will be added during processing)
        tags.extend([f'confidence_{int(confidence*100)}', scene_class])
        
        return tags[:8]  # Return max 8 tags

    def get_dominant_color(self, frame):
        """Extract dominant color from frame"""
        import numpy as np
        
        try:
            pixels = frame.reshape(-1, 3)
            avg_color = np.mean(pixels, axis=0)
            return [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]
        except:
            return [128, 128, 128]  # Default gray