#!/usr/bin/python3
import torch
import torch.nn as nn
import cv2
import numpy as np

# Define or import PointNet here
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 80 * 45, 64)  # Adjusted the dimensions here
        self.fc2 = nn.Linear(64, 12)  # Changed from 2 to 12

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PointNet().to(device)
model.load_state_dict(torch.load('/home/nvidia/fianl-project/point_predictor_epoch_70.pth'))  # Using the updated path
model.eval()

image_path = "/home/nvidia/fianl-project/data3/0.png"  # Replace with the correct path
original_img = cv2.imread(image_path)
resized_img = cv2.resize(original_img, (320, 180))

img = torch.tensor(resized_img).float().permute(2, 0, 1) / 255.0
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)

predicted_points = output.cpu().numpy()[0] * 4  # Convert the tensor output to numpy and adjust to original size

# Draw predicted points on the original image
for i in range(0, len(predicted_points), 2):
    x, y = int(predicted_points[i]), int(predicted_points[i + 1])
    cv2.circle(original_img, (x, y), 1, (0, 0, 255), -1)  # Drawing a red circle for the point

print(predicted_points)
cv2.imwrite("/home/nvidia/fianl-project/result.png", cv2.resize(original_img,(1280,720)))
