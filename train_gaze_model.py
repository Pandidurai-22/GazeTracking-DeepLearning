import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
import json
import random
import platform
import argparse
from datetime import datetime

class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        # Use ResNet101 as base model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        # Modify the final layer for gaze estimation (2D coordinates)
        self.model.fc = nn.Linear(2048, 2)
        
    def forward(self, x):
        return self.model(x)

class CustomGazeDataset(Dataset):
    def __init__(self, data_root, transform=None, system="Windows"):
        self.data_root = data_root
        self.transform = transform
        self.system = system
        self.samples = []
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Define the 9 points and their normalized coordinates
        self.point_coords = {
            'top_left': (-1, -1),
            'top_center': (0, -1),
            'top_right': (1, -1),
            'middle_left': (-1, 0),
            'middle_center': (0, 0),
            'middle_right': (1, 0),
            'bottom_left': (-1, 1),
            'bottom_center': (0, 1),
            'bottom_right': (1, 1)
        }
        
        # Load all person directories
        person_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
                
            for person in tqdm(person_dirs, desc="Loading dataset"):
                person_path = os.path.join(data_root, person)
                
                # Process each gaze direction
                for point_name, coords in self.point_coords.items():
                    point_images = [f for f in os.listdir(person_path) 
                                  if f.startswith(point_name) and f.endswith(('.jpg', '.png'))]
                    
                    for img_name in point_images:
                        img_path = os.path.join(person_path, img_name)
                        try:
                            image = cv2.imread(img_path)
                            if image is None:
                                print(f"Warning: Could not read image {img_path}")
                                continue
                                
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(image_rgb)
                            
                            if results.multi_face_landmarks:
                                landmarks = results.multi_face_landmarks[0]
                                left_eye = self.extract_eye_region(image, landmarks, True)
                                right_eye = self.extract_eye_region(image, landmarks, False)
                                
                                if left_eye is not None and right_eye is not None:
                                    self.samples.append({
                                        'left_eye': left_eye,
                                        'right_eye': right_eye,
                                        'gaze_coords': coords
                                    })
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
                            
        print(f"Loaded {len(self.samples)} samples from {len(person_dirs)} persons")
    
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        left_eye = Image.fromarray(sample['left_eye'])
        right_eye = Image.fromarray(sample['right_eye'])
        
        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)
        
        # Convert gaze coordinates to tensor
        gaze = torch.tensor(sample['gaze_coords'], dtype=torch.float32)
        
        return (left_eye, right_eye), gaze

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', checkpoint_dir=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    # Create checkpoint directory if it doesn't exist
    if checkpoint_dir is None:
        checkpoint_dir = f"model_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for (left_eyes, right_eyes), gazes in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            left_eyes = left_eyes.to(device)
            right_eyes = right_eyes.to(device)
            gazes = gazes.to(device)
            
            # Forward pass for both eyes
            left_preds = model(left_eyes)
            right_preds = model(right_eyes)
            
            # Average predictions from both eyes
            preds = (left_preds + right_preds) / 2
            
            loss = criterion(preds, gazes)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for (left_eyes, right_eyes), gazes in val_loader:
                left_eyes = left_eyes.to(device)
                right_eyes = right_eyes.to(device)
                gazes = gazes.to(device)
                
                left_preds = model(left_eyes)
                right_preds = model(right_eyes)
                preds = (left_preds + right_preds) / 2
                
                loss = criterion(preds, gazes)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss
            }, 'best_custom_gaze_model.pth')
            print("Saved best model!")

def main():
    parser = argparse.ArgumentParser(description='Train gaze estimation model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Set device based on platform
    system = platform.system()
    if system == "Windows":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # Linux/Raspberry Pi
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = CustomGazeDataset(args.data_root, transform=transform, system=system)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0 if system != "Windows" else 2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0 if system != "Windows" else 2)
    
    # Initialize model
    model = GazeEstimationModel().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, checkpoint_dir=args.checkpoint_dir)



def main():
    
    data_root = 'F:\pandidurai32mins'  # <-- Update this path to your dataset location
    
    # Define transformations
    your_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the expected size
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Create dataset
    dataset = CustomGazeDataset(data_root=data_root, transform=your_transforms)
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = GazeEstimationModel()

    # Train the model
    train_model(model, train_loader, val_loader=None, num_epochs=50, device='cuda')  # Set val_loader if you have a validation dataset



if __name__ == "__main__":
    main()


