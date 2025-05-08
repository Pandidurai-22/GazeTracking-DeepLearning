import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import time
import matplotlib.pyplot as plt
from mss import mss
import torch.nn as nn
import torchvision.transforms as transforms
import seaborn as sns

class GazeHeatmapTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize screen capture
        self.sct = mss()
        
        # Initialize gaze data storage
        self.gaze_points = []
        self.timestamps = []
        
        # Screen dimensions (adjust according to your screen)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Heatmap resolution
        self.heatmap_resolution = (192, 108)  # Scaled down for better visualization
        
    def capture_screen(self):
        """Capture the current screen content"""
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        return np.array(screenshot)
        
    def process_frame(self, frame):
        """Process a single frame and return gaze coordinates"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        return self.calculate_gaze_point(landmarks, frame.shape)
        
    def calculate_gaze_point(self, landmarks, frame_shape):
        """Calculate gaze point from facial landmarks"""
        # Eye landmarks (using MediaPipe Face Mesh indices)
        LEFT_EYE = [362, 385, 387, 263, 373]
        RIGHT_EYE = [33, 160, 158, 133, 153]
        
        def get_eye_center(eye_indices):
            points = np.array([
                [landmarks.landmark[idx].x * frame_shape[1],
                 landmarks.landmark[idx].y * frame_shape[0]]
                for idx in eye_indices
            ])
            return np.mean(points, axis=0)
        
        left_eye_center = get_eye_center(LEFT_EYE)
        right_eye_center = get_eye_center(RIGHT_EYE)
        
        # Calculate gaze point (simplified for demo)
        gaze_x = int((left_eye_center[0] + right_eye_center[0]) / 2)
        gaze_y = int((left_eye_center[1] + right_eye_center[1]) / 2)
        
        return gaze_x, gaze_y
        
    def record_gaze_data(self, duration_seconds):
        """Record gaze data for specified duration"""
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        
        print("Starting gaze recording...")
        print("Recording in progress... Please look at your screen.")
        print(f"Time remaining: {duration_seconds} seconds")
        
        while time.time() - start_time < duration_seconds:
            # Capture webcam frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Get screen content
            screen = self.capture_screen()
            
            # Process gaze
            gaze_point = self.process_frame(frame)
            if gaze_point is not None:
                self.gaze_points.append(gaze_point)
                self.timestamps.append(time.time() - start_time)
            
            # Update time remaining every second
            remaining_time = int(duration_seconds - (time.time() - start_time))
            if remaining_time % 5 == 0:  # Update every 5 seconds
                print(f"\rTime remaining: {remaining_time} seconds...", end="", flush=True)
                
        cap.release()
        print("\nRecording complete!")
        
    def generate_heatmap(self, background_image=None):
        """Generate heatmap from recorded gaze data"""
        if not self.gaze_points:
            print("No gaze data recorded!")
            return
            
        print("Generating heatmap...")
        
        # Create heatmap data
        heatmap_data = np.zeros(self.heatmap_resolution)
        scale_x = self.heatmap_resolution[0] / self.screen_width
        scale_y = self.heatmap_resolution[1] / self.screen_height
        
        # Accumulate gaze points with gaussian spreading
        for gx, gy in self.gaze_points:
            x = int(gx * scale_x)
            y = int(gy * scale_y)
            if 0 <= x < self.heatmap_resolution[0] and 0 <= y < self.heatmap_resolution[1]:
                # Add gaussian distribution around each point
                y_indices, x_indices = np.ogrid[-15:16, -15:16]
                mask = x_indices*x_indices + y_indices*y_indices <= 15*15
                for i in range(-15, 16):
                    for j in range(-15, 16):
                        if mask[i+15, j+15]:
                            new_y, new_x = y + i, x + j
                            if (0 <= new_x < self.heatmap_resolution[0] and 
                                0 <= new_y < self.heatmap_resolution[1]):
                                distance = np.sqrt(i*i + j*j)
                                intensity = np.exp(-0.1 * distance)
                                heatmap_data[new_y, new_x] += intensity
                
        # Normalize and smooth the heatmap
        heatmap_data = cv2.GaussianBlur(heatmap_data, (15, 15), 0)
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # If we have a background image, display it first
        if background_image is not None:
            # Convert to RGB if needed
            if len(background_image.shape) == 3 and background_image.shape[2] == 4:
                background_image = cv2.cvtColor(background_image, cv2.COLOR_BGRA2RGB)
            elif len(background_image.shape) == 3 and background_image.shape[2] == 3:
                background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
                
            # Resize background to match heatmap resolution
            background_image = cv2.resize(background_image, 
                                        (self.heatmap_resolution[0], self.heatmap_resolution[1]))
            plt.imshow(background_image)
        
        # Apply heatmap overlay with custom colormap
        sns.heatmap(heatmap_data, 
                   alpha=0.7,
                   cmap='jet',
                   cbar=True,
                   cbar_kws={'label': 'Gaze Intensity'})
        
        plt.title('Gaze Heatmap Analysis')
        plt.axis('off')
        
        # Save with high DPI for better quality
        plt.savefig('lecture_heatmap.png', 
                   bbox_inches='tight', 
                   pad_inches=0, 
                   dpi=300,
                   facecolor='white',
                   edgecolor='none')
        plt.close()
        
        print("Heatmap generated and saved as 'lecture_heatmap.png'")
        print("The brighter/warmer colors indicate areas where you looked more frequently")
        
    def save_data(self, filename='gaze_data.npz'):
        """Save recorded gaze data"""
        np.savez(filename,
                 gaze_points=np.array(self.gaze_points),
                 timestamps=np.array(self.timestamps))
        print(f"Data saved to {filename}")

def main():
    # Initialize tracker
    tracker = GazeHeatmapTracker()
    
    # Record gaze data (e.g., for 30 seconds)
    recording_duration = 30  # seconds
    print(f"Recording gaze data for {recording_duration} seconds...")
    tracker.record_gaze_data(recording_duration)
    
    # Save the raw data
    tracker.save_data()
    
    # Generate and save heatmap
    # Capture final screen content for background
    final_screen = tracker.capture_screen()
    tracker.generate_heatmap(background_image=final_screen)
    print("Heatmap generated and saved as 'lecture_heatmap.png'")

if __name__ == "__main__":
    main()
