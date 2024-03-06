import cv2
import numpy as np
import os
import time

# Function to display instructions
def display_instructions():
    temp_image = image.copy()
    instructions = [
        "Instructions:",
        "Left click and drag to draw a line.",
        "Press 'u' to undo the last line.",
        "Press 'r' to redo the last undone line.",
        "Press 's' to save and leave.",
        "Press 'h' to hide/show this help."
    ]
    y0, dy = 30, 30  # Initial position and line spacing
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(temp_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow('image', temp_image)
    
    cv2.waitKey(1000)  
    # Wait for 2 second before starting to fade out # time.sleep(2)
    # Use waitKey so that you can press anykey to break it.
    
    # # Fade out effect
    # num_step=25
    # for alpha in np.linspace(1, 0, num=num_step):  # Generate 10 steps from 1 to 0
    #     faded_image = cv2.addWeighted(temp_image, alpha, image, 1 - alpha, 0)
    #     cv2.imshow('image', faded_image)
    #     cv2.waitKey(1000//num_step)  # Wait for 100 ms between frames
    
    # cv2.imshow('image', image)  # Show the original image again


# Load your image
image_path = "/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/0727-0933-0272-img00.bmp"

image = cv2.imread(image_path)
original_size = image.shape[:2]  # Original size (height, width)

# Create a blank image (black image) for drawing annotations
annotation = np.zeros_like(image)
annotation_backup = np.zeros_like(image)

# Desired display size for easier annotation
scale_factor = 1.0 # Dont move. The annotation will be resized.
new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))

# Resize image for easier annotation
image = cv2.resize(image, new_size)
image_backup = image.copy()  # Backup image for undo functionality
temp_image = image.copy()  # Temporary image for showing the line preview

# Color, thickness for the annotation, and list to store lines
color = (255, 255, 255)  # White color
thickness = 1  # Line thickness
lines = []  # Store the lines drawn
undone_lines = []  # Store the undone lines for redo functionality

# Global variable to store points and the flag for dragging
points = []
dragging = False # Flag to track if mouse was dragging to draw
leave_hint = False # Flag to record if hint info should disappear
leave_help = False # Flag to record if help info should disappear
consecutive_mode = False # Add a flag for consecutive drawing mode.

# Callback function to capture mouse events
def draw_line(event, x, y, flags, param):
    global points, image, temp_image, lines, undone_lines, \
           dragging, leave_hint, consecutive_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        leave_hint = True
        if consecutive_mode and lines:
            # Start from the last point of the last line if consecutive mode is active
            points = [lines[-1][1]]  # Last point of the last line
        else:
            points = [(x, y)]  # Reset points list with the new start

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        temp_image = image.copy()
        cv2.line(temp_image, points[0], (x, y), color, thickness=thickness)
        cv2.imshow('image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
        dragging = False
        points.append((x, y))  # Add end point
        lines.append((points[0], points[1]))  # Store the line

        cv2.line(annotation, points[0], points[1], color, thickness=thickness)
        # Update temp_image with the final line for visual feedback
        cv2.line(image, points[0], points[1], color, thickness=thickness)
        cv2.imshow('image', image)  # Show the image with the final line
        ## Also clear redo history.
        undone_lines = [] 

# Create a window and bind the callback function to the window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

# Initial display with description
cv2.imshow('image', image)

while True:
    k = cv2.waitKey(0)
    # ====== Toggle consecutive drawing mode ======
    if k == ord('c'):
        consecutive_mode = not consecutive_mode  
        print(f"[Command] Consecutive drawing mode {'enabled' if consecutive_mode else 'disabled'}.")
    # ====== Undo the last line drawn ======
    if k == ord('u'):  
        if lines:
            undone_lines.append(lines.pop())  # Move the last line to undone list
            image = image_backup.copy()  # Restore the previous state
            annotation = annotation_backup.copy()  # Restore the previous state
            for line in lines:  # Redraw remaining lines
                cv2.line(image, line[0], line[1], color, thickness=thickness)
                cv2.line(annotation, line[0], line[1], color, thickness=thickness)
            cv2.imshow('image', image)
    # ====== Redo the last undone line ======
    elif k == ord('r'):  
        if undone_lines:
            line = undone_lines.pop()  # Get the last undone line
            lines.append(line)  # Move it back to lines list
            cv2.line(image, line[0], line[1], color, thickness=thickness)
            cv2.line(annotation, line[0], line[1], color, thickness=thickness)
            cv2.imshow('image', image)
    # ====== Show/hide instructions ======
    elif k == ord('h'):  
        instruction_image = image.copy()
        instructions = [
            "Left click and drag to draw a line.",
            "Press 'u' to undo the last line.",
            "Press 'r' to redo the last undone line.",
            "Press 's' to save and leave.",
            "Press 'h' to show/hide this help.",
            "Press left mouse to start drawing."
        ]
        if leave_help == False:
            y0, dy = 30, 30  # Initial position and line spacing
            for i, line in enumerate(instructions):
                y = y0 + i * dy
                cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow('image', instruction_image)
            leave_help = True
        else:
            cv2.imshow('image', image)
            leave_help = False
    # ====== Leave and Save ======
    elif k == ord('s'):  
        break
    # ====== Exception handeling (show hint) ======
    else: 
        temp_image = image.copy()
        num_frame = 25
        leave_hint = False
        # Display the description on the temporary image
        hints = [
            "Press 'u' to undo, 'r' to redo.",
            "Press 's' to save and leave.",
        ]
        y0, dy = 30, 30  # Initial position and line spacing
        for i, line in enumerate(hints):
            y = y0 + i * dy
            cv2.putText(temp_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('image', temp_image)

        # Wait for 2 second
        for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
            if leave_hint:
                break
            if cv2.waitKey(2000//num_frame) != -1:  # Wait for 100 ms between frames
                break
             
        # Fade out effect
        for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
            if leave_hint:
                break
            faded_image = cv2.addWeighted(temp_image, alpha, image, 1 - alpha, 0)
            cv2.imshow('image', faded_image)
            if cv2.waitKey(1000//num_frame) != -1:  # Wait for 100 ms between frames
                break
        cv2.imshow('image', image)

cv2.destroyAllWindows()

# Resize the annotated image back to its original size before saving
annotated_image_resized_back = cv2.resize(image, (original_size[1], original_size[0]))

# Save the annotated image
save_path = "./demo/"
os.makedirs(save_path, exist_ok=True)
cv2.imwrite(os.path.join(save_path, 'annotation_only.jpg'), annotation)

# Optional: Combine original and annotation images for visualization
combined_image = cv2.addWeighted(image, 1, annotation, 1, 0)
cv2.imwrite(os.path.join(save_path, 'combined_image.jpg'), combined_image)