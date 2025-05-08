import cv2
import numpy as np
import mediapipe as mp
import time
import os
import platform
import json
from datetime import datetime
import torch
from PIL import Image
import torchvision.transforms as transforms
from train_gaze_model import GazeEstimationModel  # Import the gaze model

class GazeRecorder:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize device-specific camera
        self.system = platform.system()
        if self.system == "Windows":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif self.system == "Linux":  # Raspberry Pi
            self.cap = cv2.VideoCapture(0)
            # Set optimized parameters for Raspberry Pi
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load gaze model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.system == "Linux":  # Force CPU on Raspberry Pi
            self.device = torch.device("cpu")
            
        self.gaze_model = GazeEstimationModel()  # Initialize the gaze model
        model_path = 'path/to/your/saved_model.pth'  # Update with the correct path to your saved model
        self.gaze_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.gaze_model.to(self.device)
        self.gaze_model.eval()  # Set the model to evaluation mode
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"gaze_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_eye_region(self, frame, landmarks, is_left=True):
        """Extract eye region from frame"""
        LEFT_EYE = [362, 385, 387, 263, 373, 380, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161]
        
        indices = LEFT_EYE if is_left else RIGHT_EYE
        points = []
        
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append((x, y))
            
        points = np.array(points)
        
        # Get bounding box with margin
        x_min = np.min(points[:, 0]) - 30
        x_max = np.max(points[:, 0]) + 30
        y_min = np.min(points[:, 1]) - 30
        y_max = np.max(points[:, 1]) + 30
        
        # Ensure within image bounds
        x_min = max(0, x_min)
        x_max = min(frame.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(frame.shape[0], y_max)
        
        if x_min >= x_max or y_min >= y_max:
            return None
            
        eye_region = frame[y_min:y_max, x_min:x_max]
        if eye_region.size == 0:
            return None
            
        return cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    
    def get_gaze_point(self, frame, landmarks):
        """Get gaze point using the trained model"""
        left_eye = self.extract_eye_region(frame, landmarks, True)
        right_eye = self.extract_eye_region(frame, landmarks, False)
        
        if left_eye is None or right_eye is None:
            return None
            
        # Convert to PIL Image and apply transforms
        left_eye = Image.fromarray(left_eye)
        right_eye = Image.fromarray(right_eye)
        
        left_eye = self.transform(left_eye).unsqueeze(0).to(self.device)
        right_eye = self.transform(right_eye).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            left_gaze = self.gaze_model(left_eye)
            right_gaze = self.gaze_model(right_eye)
            
        # Average predictions
        gaze = (left_gaze + right_gaze) / 2
        gaze = gaze.cpu().numpy()[0]
        
        # Convert normalized coordinates back to screen coordinates
        screen_width = 1920  # Adjust based on your screen
        screen_height = 1080
        
        x = int((gaze[0] + 1) * screen_width / 2)
        y = int((gaze[1] + 1) * screen_height / 2)
        
        return (x, y)
    
    def record_gaze(self, duration_seconds=30):
        """Record gaze data for specified duration"""
        gaze_data = []
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration_seconds:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                gaze_point = self.get_gaze_point(frame, landmarks)
                
                if gaze_point is not None:
                    gaze_data.append({
                        'timestamp': elapsed_time,
                        'x': gaze_point[0],
                        'y': gaze_point[1]
                    })
                    
            # Show remaining time
            remaining = duration_seconds - int(elapsed_time)
            print(f"\rTime remaining: {remaining} seconds...", end="", flush=True)
            
            # Optimize for Raspberry Pi by adding a small delay
            if self.system == "Linux":
                time.sleep(0.01)
        
        print("\nRecording complete!")
        
        # Save gaze data
        output_file = os.path.join(self.output_dir, 'gaze_data.json')
        with open(output_file, 'w') as f:
            json.dump(gaze_data, f)
            
        print(f"Gaze data saved to: {output_file}")
        
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    recorder = GazeRecorder()
    try:
        print("Starting gaze recording...")
        recorder.record_gaze(duration_seconds=30)
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main()