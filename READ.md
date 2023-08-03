# Eye-Tracking - Identifying where the eyes focus on the screen

This project utilizes deep learning techniques to recognize and track the user's eye gaze. It can be employed across various domains such as user experience analysis, game interaction, and assistive device control.

## Eye-tracking illustration

Algorithm Introduction:
 We employ a Convolutional Neural Network (CNN) to detect the user's eye position. Geometric calculations are then applied to discern the direction the eyes are gazing at. By integrating the relative position and angle of the eyes with the camera's position, the algorithm can compute the gaze point on the screen.

How does it work?
 Eye Detection: The algorithm first identifies the user's eye position. This is primarily achieved through a CNN trained on a vast dataset with labeled eye positions.
 Gaze Calculation: Once the eyes are detected, the algorithm recognizes the pupils and uses geometric methods (not necessarily AI-based) to calculate the gaze direction.
 Focus Point Calculation: Integrating the eyes' relative position, angle, and the camera's position, the algorithm computes the user's focus point on the screen.

## Running this project:

Clone or download this project to your local machine.
Install all required libraries and dependencies. You can use pip install -r requirements.txt to install all necessary libraries.
In the root directory of the project, run python main.py to initiate the program.
Ensure you have the following libraries installed:

1. PyTorch
2. OpenCV
3. Numpy
