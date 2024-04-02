import cv2 as cv
import numpy as np

def calculate_angle(curvature):
    # Define curvature thresholds
    curvature_min = 200
    curvature_max = 2000
    
    # Check if curvature is within the bounds for linear interpolation
    if abs(curvature) <= curvature_min:
        # Curvature is below the minimum threshold
        angle = 45
    elif abs(curvature) >= curvature_max:
        # Curvature is above the maximum threshold
        angle = 0
    else:
        # Linearly interpolate the angle based on the curvature
        angle = (1 - (abs(curvature) - curvature_min) / (curvature_max - curvature_min)) * 45
    
    # Adjust angle based on the sign of the curvature
    angle *= np.sign(curvature)
    
    return angle

def draw_compass(img, curvature, position=None):
    if position is None:
        # Positioning at the mid-top of the frame, slightly above for visibility
        position = (img.shape[1] // 2, 50)

    compass_radius = 100
    line_thickness = 4
    compass_color = (255, 255, 255)  # White color for half-circle and line
    arm_color = (0, 0, 255)  # Red color for the compass arm

    # Calculate the angle for the compass arm
    angle = calculate_angle(curvature)
    
    # Calculate the end point of the compass arm
    end_x = int(position[0] + compass_radius * np.sin(np.radians(angle)))
    end_y = int(position[1] - compass_radius * np.cos(np.radians(angle)))

    # Draw the half-circle and base line for the compass
    cv.ellipse(img, position, (compass_radius, compass_radius), 180, 0, 180, compass_color, thickness=line_thickness)
    cv.line(img, (position[0] - compass_radius, position[1]), (position[0] + compass_radius, position[1]), compass_color, line_thickness)

    # Draw the compass arm
    cv.line(img, position, (end_x, end_y), arm_color, line_thickness)

    return img
