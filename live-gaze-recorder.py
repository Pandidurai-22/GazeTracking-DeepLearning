import cv2
import numpy as np
import pygame
import torch
from torchvision import transforms  # Ensure this import is included
from torchvision.models import resnet101, ResNet101_Weights
from calibrated_gaze_tracker import CalibratedGazeTracker

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CalibratedGazeTracker:
    def __init__(self, model):
        self.gaze_model = model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def get_gaze_point(self, frame):
        input_tensor = self.transform(frame)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = self.gaze_model(input_tensor)
            
        print("Output shape:", output.shape)  # Debugging line
        if output.dim() == 1:
            x, y = output[0].item(), output[1].item()  # If output is [2]
        elif output.dim() == 2 and output.size(0) == 1:
            x, y = output[0, 0].item(), output[0, 1].item()  # If output is [1, 2]
        else:
            raise ValueError("Unexpected output shape from the model")
            
        return x, y

class GazeEstimationModel(torch.nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        # Use ResNet101 as base model
        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Linear(2048, 2)  # Adjust the output layer for gaze estimation

    def forward(self, x):
        return self.model(x)

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption('Live Gaze Tracker')

# Bubble color
bubble_color = (255, 0, 0)

# Function to draw the bubble indicator
def draw_bubble(x, y):
    pygame.draw.circle(screen, bubble_color, (int(x), int(y)), 20)

# Function to retrieve gaze coordinates from the gaze tracking model
def get_gaze_coordinates():
    ret, frame = camera.read()  # Get frame from camera
    if not ret:
        return (0, 0)  # Return default coordinates if frame capture fails

    # Use the gaze tracker to get the gaze point
    gaze_point = tracker.get_gaze_point(frame)  # Assuming tracker is an instance of CalibratedGazeTracker
    return gaze_point if gaze_point else (0, 0)

# Initialize the camera and gaze prediction model
camera = cv2.VideoCapture(0)  # Replace with your camera setup
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

pg = GazeEstimationModel().to(device)
tracker = CalibratedGazeTracker(pg)

# Main loop for gaze tracking
running = True
tracking = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                tracking = False
            if event.key == pygame.K_RETURN:
                tracking = True

    if tracking:
        # Get the gaze coordinates (x, y) from your gaze tracking model
        x, y = get_gaze_coordinates()

        x= max(0, min(x, 1920))
        y= max(0, min(y, 1080))

        screen.fill((0, 0, 0))  # Clear the screen
        draw_bubble(x, y)  # Draw the bubble
    else:
        # Optionally, display a message to start tracking
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render('Press ENTER to start tracking', True, (255, 255, 255))
        screen.blit(text, (50, 50))

    pygame.display.flip()  # Update the display
    pygame.time.delay(100)

pygame.quit()