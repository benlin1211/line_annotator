import cv2
import numpy as np
import os

# Load your image
image_path = "/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/0727-0933-0272-img00.bmp"

image = cv2.imread(image_path)
backup_image = image.copy()  # Backup image for undo functionality
temp_image = image.copy()  # Temporary image for showing the line preview

# Color, thickness for the annotation, and list to store lines
color = (255, 255, 255)  # White color
thickness = 1  # Line thickness
lines = []  # Store the lines drawn
undone_lines = []  # Store the undone lines for redo functionality

# Global variable to store points and the flag for dragging
points = []
dragging = False

# Callback function to capture mouse events
def draw_line(event, x, y, flags, param):
    global points, image, temp_image, dragging, lines, undone_lines

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        points = [(x, y)]  # Reset points list with the new start

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        temp_image = image.copy()
        cv2.line(temp_image, points[0], (x, y), color, thickness=thickness)
        # Display the description on the temporary image
        cv2.putText(temp_image, "Press 'u' to undo, any other key to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('image', temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        points.append((x, y))  # Add end point
        lines.append((points[0], points[1]))  # Store the line
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
    if k == ord('u'):  # Undo the last line drawn
        if lines:
            undone_lines.append(lines.pop())  # Move the last line to undone list
            image = backup_image.copy()  # Restore the previous state
            for line in lines:  # Redraw remaining lines
                cv2.line(image, line[0], line[1], color, thickness=thickness)
            cv2.imshow('image', image)
    elif k == ord('r'):  # Redo the last undone line
        if undone_lines:
            line = undone_lines.pop()  # Get the last undone line
            lines.append(line)  # Move it back to lines list
            cv2.line(image, line[0], line[1], color, thickness=thickness)
            cv2.imshow('image', image)
    else:
        break

cv2.destroyAllWindows()

# Save the annotated image
save_path = "./demo/"
os.makedirs(save_path, exist_ok=True)
cv2.imwrite(os.path.join(save_path,'annotated_image.jpg'), image)
