import cv2
import numpy as np
import os
import time


class AnnotatorState():
    def __init__(self) -> None:
        self.dragging = False # Flag to track if mouse was dragging to draw
        self.leave_hint = False # Flag to record if hint info should disappear
        self.leave_help = False # Flag to record if help info should disappear
        self.consecutive_mode = False # Add a flag for consecutive drawing mode.
        self.drawing_mode = "line"

    ## TODO...

def show_help(info_list: list):
    for line in info_list:
        print(line)
    # instruction_image = image.copy()
    # if state.leave_help == False:
    #     y0, dy = 30, 30  # Initial position and line spacing
    #     for i, line in enumerate(info_list):
    #         y = y0 + i * dy
    #         cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    #     cv2.imshow('image', instruction_image)
    #     state.leave_help = True
    # else:
    #     cv2.imshow('image', image)
    #     state.leave_help = False

def show_hint(info_list, image, state, duration=2000, num_frame=25):
    # Show on cmd.
    show_help(info_list)
    temp_image = image.copy()
    state.leave_hint = False
    y0, dy = 30, 30  # Initial position and line spacing
    for i, line in enumerate(info_list):
        y = y0 + i * dy
        cv2.putText(temp_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow('image', temp_image)
    # Wait for 2 second
    for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
        if state.leave_hint:
            break
        elif cv2.waitKey(duration//num_frame) != -1:  # Wait for 100 ms between frames
            state.leave_hint=True
        
    # Fade out effect
    for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
        if state.leave_hint:
            break
        faded_image = cv2.addWeighted(temp_image, alpha, image, 1 - alpha, 0)
        cv2.imshow('image', faded_image)
        if cv2.waitKey(1000//num_frame) != -1:  # Wait for 100 ms between frames
            state.leave_hint=True
    cv2.imshow('image', image)
     
if __name__=="__main__":
    # Load your image
    image_path = "/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/0727-0933-0272-img00.bmp"
    annotation_path = "/home/pywu/Downloads/zhong/line_annotator/UV-only/0727-0933-0272-img00_UV.bmp"  # Change to your actual path

    image = cv2.imread(image_path)

    ## Already in opencv-python==4.9.0.80
    # Desired display size for easier annotation
    scale_factor = 1.0 # Dont move. The annotation will be resized.
    assert scale_factor==1.0
    original_size = image.shape[:2]  # Original size (height, width)
    new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
    image = cv2.resize(image, new_size)

    # Load previous annotations
    if os.path.exists(annotation_path):
        print(f"[INFO] Load existing annotation from {annotation_path}")
        annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        if annotation.ndim == 2 or annotation.shape[2] == 1:  # If the loaded annotation is grayscale
            annotation = cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)
        # Plot previous annotations on image. 
        image = cv2.addWeighted(image, 1, annotation, 0.5, 0)
    else:
        # Create a blank image (black image) for drawing annotations
        annotation = np.zeros_like(image)

    # Create Backups f
    temp_image = image.copy()  # Temporary image for showing the line preview
    image_backup = image.copy()  # Backup image for undo functionality
    annotation_backup = annotation.copy() # Backup image for undo functionality


    # Color, thickness for the annotation, and list to store lines
    color = (255, 255, 255)  # White color
    thickness = 1  # Line thickness
    lines = []  # Store the lines drawn
    undone_lines = []  # Store the undone lines for redo functionality

    # Global variable to store points and the flag for dragging
    points = []
    state = AnnotatorState()

    # Callback function to capture mouse events
    def draw_line(event, x, y, flags, param):
        global points, image, temp_image, lines, undone_lines, state
            # dragging, leave_hint, consecutive_mode
        if state.drawing_mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                state.dragging = True
                state.leave_hint = True
                if state.consecutive_mode and lines:
                    # Start from the last point of the last line if consecutive mode is active
                    points = [lines[-1][1]]  # Last point of the last line
                else:
                    points = [(x, y)]  # Reset points list with the new start

            elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
                temp_image = image.copy()
                cv2.line(temp_image, points[0], (x, y), color, thickness=thickness)
                cv2.imshow('image', temp_image)

            elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
                state.dragging = False
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
        # ====== Switch drawing mode ======
        if k == ord('x'):
            state.drawing_mode = "scratch" if state.drawing_mode == "line" else "line"
            hints = [f"Switched to {state.drawing_mode} mode."]
            show_hint(hints, image, state)            
        # ====== Toggle consecutive drawing mode ======
        elif k == ord('c'):
            if state.drawing_mode == "line":
                state.consecutive_mode = not state.consecutive_mode  
                hints = [f"Consecutive drawing mode {'enabled' if state.consecutive_mode else 'disabled'}."]
                show_hint(hints, image, state)
            else:
                hints = [f"Consecutive drawing is avaliable under drawing_mode only."]
                show_help(hints)
        # ====== Undo the last line drawn ======
        elif k == ord('u'):  
            if lines:
                undone_lines.append(lines.pop())  # Move the last line to undone list
                image = image_backup.copy()  # Restore the previous state
                annotation = annotation_backup.copy()  # Restore the previous state
                for line in lines:  # Redraw remaining lines
                    cv2.line(image, line[0], line[1], color, thickness=thickness)
                    cv2.line(annotation, line[0], line[1], color, thickness=thickness)
                cv2.imshow('image', image)
                print("[INFO] Undo.")
        # ====== Redo the last undone line ======
        elif k == ord('r'):  
            if undone_lines:
                line = undone_lines.pop()  # Get the last undone line
                lines.append(line)  # Move it back to lines list
                cv2.line(image, line[0], line[1], color, thickness=thickness)
                cv2.line(annotation, line[0], line[1], color, thickness=thickness)
                cv2.imshow('image', image)
                print("[INFO] Redo.") 
        # ====== Show/hide instructions ======
        elif k == ord('h'):  
            instruction_image = image.copy()
            instructions = [
                "============= Welcome to Line Annotator! =============",
                "Press left click to draw a line.",
                "Undo: 'u'",
                "Redo: 'r'",
                "Switch to Line mode: 'x'",
                "Leave and Save: 's'",
                "Leave without Saveing: 'esc'",
                "=============== Author: Zhong-Wei Lin ===============", 
                "",
            ]
            show_help(instructions)
            # if state.leave_help == False:
            #     y0, dy = 30, 30  # Initial position and line spacing
            #     for i, line in enumerate(instructions):
            #         y = y0 + i * dy
            #         cv2.putText(instruction_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            #     cv2.imshow('image', instruction_image)
            #     state.leave_help = True
            # else:
            #     cv2.imshow('image', image)
            #     state.leave_help = False
        # ====== Leave and Save ======
        elif k == ord('s'):  
            # Resize the annotated image back to its original size before saving
            annotated_image_resized_back = cv2.resize(image, (original_size[1], original_size[0]))

            # Save the annotated image
            save_path = "./demo/"
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'annotation_only.jpg'), annotation)

            # Optional: Combine original and annotation images for visualization
            combined_image = cv2.addWeighted(image, 1, annotation, 1, 0)
            cv2.imwrite(os.path.join(save_path, 'combined_image.jpg'), combined_image)
            print("[INFO] Save results at save_path.")
            break
        # ====== ESC key to leave without saving ======
        elif k == 27: 
            print("[INFO] Leave without saving.")
            break
        # ====== Exception handeling (show hint) ======
        else: 
            temp_image = image.copy()
            # Display the description on the temporary image
            hints = [
                "Hint:",
                "Press 'u' to undo, 'r' to redo.",
                "Press 's' to save and leave.",
            ]
            show_hint(hints, image, state)
            

    cv2.destroyAllWindows()

