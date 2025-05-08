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
from collections import deque
import matplotlib.pyplot as plt

if platform.system() == "Windows":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchvision.models import resnet101, ResNet101_Weights
else:
    from PIL import ImageDraw
    from PIL import Image as PILImage
    import picamera2
    from picamera2 import Picamera2
    import torch

    from torchvision.models import resnet101, ResNet101_Weights

class AdvancedCalibrationManager:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        self.user_3d_model = None
        self.facial_measurements = {}
        self.translation_history = deque(maxlen=30)
        self.last_recalibration = time.time()
        self.calibration_interval = 300  # 5 minutes
        self.reference_z_distance = 600  # 60cm reference distance

    def calibrate_camera(self, checkerboard_images):
        """Calibrate camera using checkerboard pattern"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 25  # 25mm squares
        
        objpoints, imgpoints = [], []
        for img in checkerboard_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
        
        ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        self.calibrated = ret

    def measure_facial_geometry(self, landmarks):
        """Calculate user-specific facial dimensions"""
        # Extract key facial measurements
        left_eye = np.mean([landmarks[33], landmarks[133]], axis=0)
        right_eye = np.mean([landmarks[362], landmarks[263]], axis=0)
        nose_tip = landmarks[1]
        
        self.facial_measurements['interocular'] = np.linalg.norm(right_eye - left_eye)
        self.facial_measurements['eye_nose_dist'] = np.mean([
            np.linalg.norm(left_eye - nose_tip),
            np.linalg.norm(right_eye - nose_tip)
        ])
        
        self._create_personalized_3d_model()

    def _create_personalized_3d_model(self):
        """Create user-specific 3D model based on measurements"""
        base_model = np.array([
            [-112.5, -125, -135],   # Left eye outer
            [112.5, -125, -135],    # Right eye outer
            [0, 0, 0],              # Nose tip
            [-70, -125, -135],      # Left eye inner
            [70, -125, -135],       # Right eye inner
            [0, -90, -135],         # Nose bridge
        ], dtype=np.float32)
        
        # Scale based on measurements
        scale_factor = self.facial_measurements['interocular'] / 225
        self.user_3d_model = base_model * scale_factor

    def update_translation(self, tvec):
        """Track head position for compensation"""
        self.translation_history.append(tvec)
        if len(self.translation_history) >= 10:
            return np.mean(self.translation_history, axis=0)
        return tvec

    def check_recalibration_needed(self):
        """Check if periodic recalibration is needed"""
        return time.time() - self.last_recalibration > self.calibration_interval

    def apply_calibration(self, gaze_vector, head_pose):
        """Apply full calibration chain to gaze estimation"""
        if not self.calibrated:
            return gaze_vector, head_pose
            
        # Apply translation compensation
        compensated_pose = self._compensate_translation(head_pose)
        
        # Apply user-specific adjustments
        if self.user_3d_model is not None:
            gaze_vector = self._adjust_for_facial_geometry(gaze_vector)
            
        return gaze_vector, compensated_pose

    def _compensate_translation(self, head_pose):
        """Compensate for head movement effects"""
        if len(self.translation_history) < 10:
            return head_pose
            
        avg_translation = np.mean(self.translation_history, axis=0)
        z_scale = avg_translation[2] / self.reference_z_distance
        
        # Adjust angles based on relative position
        pitch, yaw = head_pose
        pitch *= z_scale
        yaw *= z_scale
        
        return (pitch, yaw)

    def _adjust_for_facial_geometry(self, gaze_vector):
        """Adjust gaze vector based on personal facial geometry"""
        if not self.facial_measurements:
            return gaze_vector
            
        scale = self.facial_measurements['interocular'] / 225
        return gaze_vector * scale

class GazeEstimationModel(torch.nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        # Use ResNet50 as base model
        if platform.system() == "Windows":
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
            self.model.fc = torch.nn.Linear(2048, 2)
        else:
            self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            self.model.fc = torch.nn.Linear(2048, 2)

    def forward(self, x):
        return self.model(x)

class CalibratedGazeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize platform-specific camera and settings
        self.system = platform.system()
        if self.system == "Windows":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.screen_width = 1920
            self.screen_height = 1080
        else:  # Linux/Raspberry Pi
            # Initialize Picamera2 properly
            self.cap = Picamera2()
            preview_config = self.cap.create_preview_configuration(main={"size": (1920, 1080)}) #640,480
            self.cap.configure(preview_config)
            self.cap.start()
            
            self.screen_width = 1920 #1920
            self.screen_height = 1080 #1080
        
        # Initialize gaze model with platform-specific settings
        if self.system == "Windows":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # Force CPU on Linux/Raspberry Pi
            self.device = torch.device("cpu")
            
        self.gaze_model = GazeEstimationModel()
        
        # Load model weights
        custom_model_path = 'best_mpiigaze_model.pth'
        eth_model_path = 'resnet101-5d3b4d8f.pth'

        if os.path.exists(custom_model_path):
            checkpoint = torch.load(custom_model_path, map_location=self.device)
            self.gaze_model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded custom trained gaze model!")
        elif os.path.exists(eth_model_path):
            state_dict = torch.load(eth_model_path, map_location=self.device, weights_only=False)
            if 'model' in state_dict:
                self.gaze_model.model.load_state_dict(state_dict['model'])
                print("Loaded ETH-XGaze pretrained model!")

        self.gaze_model.to(self.device)
        self.gaze_model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Calibration points (normalized coordinates)
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        
        self.calibration_data = {}
        self.ref_points = None
        
        # Create output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"gaze_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Add advanced calibration manager
        self.advanced_calibration = AdvancedCalibrationManager()
        
    def get_frame(self):
        """Get frame from camera based on platform"""
        if self.system == "Windows":
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        else:  # Linux/Raspberry Pi
            try:
                frame = self.cap.capture_array()
                return frame
            except Exception as e:
                print(f"Error capturing frame: {e}")
                return None
    
    def get_background_path(self):
        """Get system-specific background image path"""
        if self.system == "Windows":
            return "solid-new.jpg"
        else:  # Linux/Raspberry Pi
            # First try the home directory
            home_path = os.path.expanduser("~")
            paths_to_try = [
                os.path.join(home_path, "solid-new.jpeg"),
                os.path.join(os.getcwd(), "solid-black.jpg"),
                os.path.join(os.getcwd(), "solid-new.jpeg")
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    return path
                    
            # If no background image found, return the Windows default
            return "solid-new.jpg"
    
    def calibrate(self):
        """Perform full calibration sequence"""
        print("Starting calibration sequence...")
        print("Please look at each point as it appears")


        # for point in self.calibration_points:
        #     x_pixel = int(point[0] * self.screen_width)
        #     y_pixel = int(point[1] * self.screen_height)
            
        #     while True:
        #         frame = self.get_frame()  
        #         if frame is None:
        #             continue
                
        #         gaze_point = self.get_gaze_point(frame)
        #         if gaze_point is not None:
        #             cv2.circle(frame, (int(gaze_point[0]), int(gaze_point[1])), 10, (0, 0, 255), -1)  # Red circle
        #             cv2.circle(frame, (x_pixel, y_pixel), 20, (0, 255, 0), -1)  # Green circle for calibration point
        #             # Show the frame with the red dot and calibration point
        #             cv2.imshow("Calibration", frame)
        #             # Check if the user is looking at the point (you can implement your logic here)
        #             if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to move to the next point
        #                 break
        # cv2.destroyAllWindows()
    
        
        background_path = self.get_background_path()
        background = cv2.imread(background_path)
        if background is None:
            print("Creating black background as no background image found")
            background = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        window_name = 'Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Collect facial measurements during calibration
        frame = self.get_frame()
        if frame is not None:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1]] 
                                    for lm in results.multi_face_landmarks[0].landmark])
                self.advanced_calibration.measure_facial_geometry(landmarks)
        
        for i, (rel_x, rel_y) in enumerate(self.calibration_points):
            x = int(rel_x * self.screen_width)
            y = int(rel_y * self.screen_height)
            
            display = background.copy()
            cv2.circle(display, (x, y), 20, (0, 255, 0), -1)
            
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
            
            start_time = time.time()
            eye_features = []
            
            while time.time() - start_time < 2:
                frame = self.get_frame()
                if frame is None:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_eye = self.extract_eye_region(frame, landmarks, True)
                    right_eye = self.extract_eye_region(frame, landmarks, False)
                    
                    if left_eye is not None and right_eye is not None:
                        eye_features.append({
                            'left_eye': left_eye,
                            'right_eye': right_eye
                        })
                        
                        # Update translation tracking
                        if hasattr(results, 'pose_landmarks'):
                            nose_tip = results.pose_landmarks.landmark[0]
                            tvec = np.array([nose_tip.x, nose_tip.y, nose_tip.z])
                            self.advanced_calibration.update_translation(tvec)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            # Store calibration data
            if eye_features:
                self.calibration_data[f'point_{i}'] = {
                    'position': (x, y),
                    'features': eye_features
                }
        
        cv2.destroyWindow(window_name)
        self.advanced_calibration.last_recalibration = time.time()
    
    def save_calibration_data(self):
        """Save calibration data"""
        calibration_file = os.path.join(self.output_dir, 'calibration_data.json')
        
        # Convert eye features to list for JSON serialization
        serializable_data = {}
        for point_idx, data in self.calibration_data.items():
            serializable_data[str(point_idx)] = {
                'position': data['position'],
                'timestamp': datetime.now().isoformat()
            }
        
        with open(calibration_file, 'w') as f:
            json.dump(serializable_data, f)
    
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
        
        # file save

        landmark_data = {'left_eye1': landmark[0], 'right_eye': landmark[1]}
        landmark_file_path = os.path.join(self.output_dir, 'facial_landmark_data.json')
        with open(landmark_file_path, 'w') as f:
            json.dump(landmark_data, f)
            print(f"Facial landmark data saved to {landmark_file_path}")
    
    def get_gaze_point(self, frame):
        """Get calibrated gaze point from frame"""
        if not self.calibration_data:
            print("Error: Calibration required before tracking")
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        left_eye = self.extract_eye_region(frame, landmarks, True)
        right_eye = self.extract_eye_region(frame, landmarks, False)
        
        if left_eye is not None and right_eye is not None:
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
            
            # Apply advanced calibration
            gaze, head_pose = self.advanced_calibration.apply_calibration(gaze, (0, 0))
            
            # Convert normalized coordinates back to screen coordinates
            x = int(np.clip((-gaze[0] + 1) / 2 * self.screen_width, 0, self.screen_width - 1))
            y = int(np.clip((-gaze[1] + 1) / 2 * self.screen_height, 0, self.screen_height - 1))
            
            return x, y
    
    def record_gaze_data(self, duration_seconds):
        """Record gaze data for specified duration"""
        if not self.calibration_data:
            print("Please run calibration first!")
            return
            
        print(f"\nStarting gaze recording for {duration_seconds} seconds...")
        print("Please look at your screen some areas. now heatmap generation process going on.. :]")
        
        gaze_data = []
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration_seconds:
                break
                
            frame = self.get_frame()
            if frame is None:
                continue
                
            gaze_point = self.get_gaze_point(frame)
            if gaze_point is not None:
                gaze_data.append({
                    'timestamp': elapsed_time,
                    'x': gaze_point[0],
                    'y': gaze_point[1]
                })
                
            remaining = duration_seconds - int(elapsed_time)
            print(f"\rTime remaining: {remaining} seconds...", end="", flush=True)
            
            # Add small delay for Raspberry Pi
            if self.system != "Windows":
                time.sleep(0.01)
        
        print("\nRecording complete!")
        
        # Save gaze data
        output_file = os.path.join(self.output_dir, 'gaze_data.json')
        with open(output_file, 'w') as f:
            json.dump(gaze_data, f)
            
        print("Generating heatmap...")
        self.generate_heatmap(gaze_data)
        
    def generate_heatmap(self, gaze_data):
        heatmap = np.zeros((self.screen_height, self.screen_width))
        
        # for entry in gaze_data:
        #     x= entry['x']
        #     y= entry['y']
        #     x = min(max(int(x), 0), self.screen_width - 1)
        #     y = min(max(int(y), 0), self.screen_height - 1)
        #     heatmap[y, x] += 1  # Increment heatmap value
        
        kernel_size = 50  # Size of the gaussian kernel
        sigma = 25  # Standard deviation
        for gaze in gaze_data:
            x, y = int(gaze['x']), int(gaze['y'])
            if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
                y_grid, x_grid = np.ogrid[-kernel_size:kernel_size + 1, -kernel_size:kernel_size + 1]
                gaussian_kernel = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2*sigma*sigma))
                
                y_min = max(0, y - kernel_size)
                y_max = min(self.screen_height, y + kernel_size + 1)
                x_min = max(0, x - kernel_size)
                x_max = min(self.screen_width, x + kernel_size + 1)
                
                heatmap[y_min:y_max, x_min:x_max] += gaussian_kernel[:y_max-y_min, :x_max-x_min]

        # Normalize and apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_COOL)

        num_top_points = 3
        top_indices = np.argsort(heatmap.flatten())[-num_top_points:]
        top_points = np.unravel_index(top_indices, heatmap.shape)

        for point in zip(*top_points):
            cv2.circle(heatmap_color, (point[1],point[0]), 20, (255,255,255), -1)

        if platform.system() == "Windows":
            # Visualization
            plt.figure(figsize=(16, 9))
            # sns.heatmap(heatmap, cmap='jet', alpha=0.6)
            plt.imshow(heatmap_color)
            plt.axis('off')
            output_path = os.path.join(self.output_dir, 'gaze_heatmap.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Heatmap saved as: {output_path}")

        else:
            plt.figure(figsize=(16, 9))
            plt.imshow(heatmap, cmap='jet', alpha=0.6, origin='upper')
            plt.axis('off')
            output_path = os.path.join(self.output_dir, 'gaze_heatmap.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Heatmap saved as: {output_path}")
        # else:
        #     heatmap_image = PILImage.new("RGB", (self.screen_width, self.screen_height))
        #     draw = ImageDraw.Draw(heatmap_image)
        #     for (x, y) in np.argwhere(heatmap > 0):
        #         draw.point((x,y), fill=(255, 0, 0, int(heatmap[y, x]* 255)))
        #     output_path = os.path.join(self.output_dir, 'gaze_heatmap.png')
        #     heatmap_image.save(output_path)
        #     print(f"Heatmap saved as: {output_path}")


    def log_session_details(self, message):
        log_file_path = os.path.join(self.output_dir, 'session_log.txt')
        with open(log_file_path, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def cleanup(self):
        """Release resources"""
        if self.system == "Windows":
            self.cap.release()
        else:
            self.cap.stop()
        cv2.destroyAllWindows()

def main():
    tracker = CalibratedGazeTracker()
    try:
        tracker.calibrate()
        tracker.record_gaze_data(duration_seconds=30)
        tracker.save_calibration_data()
        tracker.log_session_details("log session completed")
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()
