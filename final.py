#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import gc

class RecognitionDataset(Dataset):
    def __init__(self, image_paths, points_list):
        self.image_paths = image_paths
        self.points_list = points_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (320, 180))
        img = torch.tensor(img).float().permute(2, 0, 1) / 255.0
        
        points_data = self.points_list[idx]
        adjusted_points = []
        for group in points_data:
            for point in group:
                if isinstance(point, tuple):
                    adjusted_points.append((point[0]/4, point[1]/4))
                else:
                    for coord in point:
                        adjusted_points.append((coord[0]/4, coord[1]/4))
        
        label = [coord for point in adjusted_points for coord in point]
        
        label = torch.tensor(label).float()
        return img, label

def read_data_from_path(data_path):
    with open(data_path + 'points.txt', 'r') as file:
        lines = file.readlines()
    
    image_paths = [data_path + "pictures/" + line.strip().split(':')[0] for line in lines]
    points_list = [eval(line.strip().split(':')[1]) for line in lines]
    
    return image_paths, points_list

image_paths_1, points_list_1 = read_data_from_path('/home/nvidia/fianl-project/data/')
image_paths_2, points_list_2 = read_data_from_path('/home/nvidia/fianl-project/data2/')
image_paths_3, points_list_3 = read_data_from_path('/home/nvidia/fianl-project/data3/')

image_paths = image_paths_1 + image_paths_2 + image_paths_3
points_list = points_list_1 + points_list_2 + points_list_3

dataset = RecognitionDataset(image_paths, points_list)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Using multiple workers to preload the data
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 80 * 45, 64)
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PointNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (epoch + 1) % 10 == 0 and epoch > 50:
        torch.save(model.state_dict(), f'/home/nvidia/fianl-project/point_predictor_epoch_{epoch+1}.pth')

torch.save(model.state_dict(), '/home/nvidia/fianl-project/point_predictor.pth')
gc.collect()
