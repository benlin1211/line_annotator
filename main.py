import cv2
import numpy as np
import os
import time
from scipy.ndimage import convolve
import glob, re
# import tkinter as tk
# from tkinter import ttk
from utils.data_selector import select_image_annotation_pair_by_index, read_image_and_annotation #, run_selector_app
from utils.printer import print_on_console, print_on_image
import threading

import tkinter as tk
from tkinter import simpledialog
import argparse


class Annotator():
    def __init__(self) -> None:
        self.color = (255, 255, 255)  # White color
        self.thickness = 1  # Line thickness

        # List to store lines
        self.lines = []  # Store the lines drawn
        self.undone_lines = []  # Store the undone lines for redo functionality

        # State variable to store the flags
        self.state = AnnotatorState()


class AnnotatorState():
    def __init__(self) -> None:
        # Color, thickness for the annotation

        self.is_dragging = False # Flag to track if mouse was is_dragging to draw
        self.is_scratching = False # Flag to track if mouse was is_scratching
        self.leave_hint = False # Flag to record if hint info should disappear
        self.leave_help = False # Flag to record if help info should disappear
        self.is_consecutive_line = None # Add a flag for consecutive drawing mode.
        self.drawing_mode = "nearest"
        
        self.immediately_draw=False

    ## TODO: show status
    def check_state(self):
        pass
    
    # is_dragging behavior
    def start_dragging(self):
        self.is_dragging = True
        self.leave_hint = True 

    def end_draggging(self):
        self.is_dragging = False

    # straching behavior
    def start_scratching(self):
        self.scratch = True
        self.leave_hint = True 

    def end_scratching(self):
        self.scratch = False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image and Annotation Loader with Custom Save Path.')
    parser.add_argument('--save_path', type=str, default='./result/', help='Path to save the output results.')
    parser.add_argument('--demo_path', type=str, default='./demo/', help='Path to save the demo.')
    parser.add_argument('--image_root', type=str, default='/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/', help='Root directory for images.')
    parser.add_argument('--annotation_root', type=str, default='/home/pywu/Downloads/zhong/dataset/teeth_qisda/supplements/0727-0933/UV-only/', help='Root directory for annotations.')
    
    args = parser.parse_args()
    return args


def read_data_pairs(image_set, annotation_set):
    annotation_set = sorted(glob.glob(os.path.join(annotation_root, "*.bmp")))
    sequence_numbers = []
    for a in annotation_set:
        number = os.path.basename(a).split("-")[-2]
        sequence_numbers.append(number)

    pattern = re.compile(r"-img0[0-2]\.bmp$")
    image_set = sorted([name for name in glob.glob(os.path.join(image_root, "*.bmp")) \
                        if (pattern.search(name) and os.path.basename(name).split("-")[-2] in sequence_numbers)])
    # for p in image_set:
    #     if  os.path.basename(p).split("-")[-2] not in sequence_numbers:
    #         print(p)
    assert len(image_set) == len(annotation_set), \
        f"Number of image ({len(image_set)}) and annotation ({len(annotation_set)}) are not identical"
    
    return image_set, annotation_set




def extract_sub_image(image, x, y, roi_dim):
    """
    Extract a sub-image centered at (x, y) with a size of 9x9 (index from [x-4: x+4, y-4:y+4]).
    Includes boundary handling to ensure the indices do not go out of the image bounds.
    
    Parameters:
    - image: The input image as a NumPy array.
    - x, y: The center coordinates of the sub-image.
    
    Returns:
    - The extracted sub-image as a NumPy array.
    """
    # Determine the shape of the input image
    height, width = image.shape[:2]
    
    # Calculate start and end indices with boundary handling
    start_x = max(x - roi_dim//2, 0)
    end_x = min(x + roi_dim//2+1, width)  # +5 because upper bound is exclusive
    start_y = max(y - roi_dim//2, 0)
    end_y = min(y + roi_dim//2+1, height)
    
    # Extract the sub-image
    sub_image = image[start_y:end_y, start_x:end_x]
    
    return sub_image


def detect_endpoints_local(ROI, target_color):
    
    binary_map = np.where(np.all(ROI == target_color, axis=-1), 1, 0)
    
    # Define the kernel for 8-connectivity
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Convolve binary map with kernel to count neighbors
    neighbor_count = convolve(binary_map.astype(np.uint8), kernel, mode='constant', cval=0)
    
    # Endpoints are segment pixels with exactly one neighbor
    endpoints = (binary_map == 1) & (neighbor_count == 1)

    # Find nonzero elements (endpoints), returns in (y, x) format
    yx_coordinates = np.transpose(np.nonzero(endpoints))

    #  Convert (y, x) to (x, y)
    xy_coordinates = yx_coordinates[:, [1, 0]]
    return xy_coordinates


def local2global(local_point, curser_pos, roi_dim, image_size):

    local_nx, local_ny = local_point
    gx, gy = curser_pos
    # assert h%2==1 and w%2==1 ## odd square

    # # Find the top-lefy corner coordinate for reference point
    inner_shift_x = min(roi_dim//2, gx)
    inner_shift_y = min(roi_dim//2, gy)
    shift_x = min(max(gx - inner_shift_x, 0), image_size[1] - 1)
    shift_y = min(max(gy - inner_shift_y, 0), image_size[0] - 1)

    global_nx = shift_x + local_nx
    global_ny = shift_y + local_ny
    global_nx = max(0, min(global_nx, image_size[1] - 1))
    global_ny = max(0, min(global_ny, image_size[0] - 1))

    return (global_nx, global_ny)


def find_nearest_point_on_map_within_range(roi_dim, local_endpoints, curser_pos, image_size, range):
    """
    Find the nearest pixel position of target_color to the given position in the image.
    """
    
    #x, y = curser_pos # global

    # assert h%2==1 and w%2==1
    # If the center of ROI is not cursor (ROI hitting image boundary)
    # Calculate distances from cursor position to all points with target color
    gx, gy = curser_pos
    inner_shift_x = min(roi_dim//2, gx)
    inner_shift_y = min(roi_dim//2, gy)

    ## TODO: find all endpoints in ROI.
    if len(local_endpoints) == 0:
        return curser_pos  # Return the original position if no target color found
    else:
        ones_x, ones_y = np.transpose(local_endpoints)
        distances = (ones_x - inner_shift_x) ** 2 + (ones_y - inner_shift_y) ** 2
        # Find the index of the minimum distance
        if np.amin(distances) > range**2:
            return curser_pos  # Return the original position
        else:
            nearest_index = np.argmin(distances)
            # Return the nearest point (note the reversal from y,x to x,y)
            local_nx = ones_x[nearest_index]
            local_ny = ones_y[nearest_index]
            ## Convert local coordinate in a 2D ROI to global coordinate value.
            global_nx, global_ny = local2global((local_nx, local_ny), curser_pos, roi_dim, image_size)
        
    return (global_nx, global_ny)


def add_semi_transparent_rectangle(image, top_left, bottom_right, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)  # -1 fills the rectangle
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def initialize_annotator(image_path, annotation_path):

    image, annotation = read_image_and_annotation(image_path, annotation_path)

    # Create Backups 
    temp_image = image.copy()  # Temporary image for showing the line preview
    image_backup = image.copy()  # Backup image for undo functionality
    annotation_backup = annotation.copy() # Backup image for undo functionality

    # Create Annotator 
    myAnn = Annotator()

    return image, annotation, temp_image, image_backup, annotation_backup, myAnn


# # This function runs the PyQt app in a separate thread
# def open_image_selector():
#     run_selector_app(image_set)

# Function to create and show the Tkinter list selection window
def open_image_selector(image_set):
    def on_select(evt):
        # Event handler for selecting an item from the list
        w = evt.widget
        index = int(w.curselection()[0])
        global current_image_index
        current_image_index = index
        top.destroy()

    top = tk.Tk()
    top.title('Select an Image')

    listbox = tk.Listbox(top, width=50, height=20)
    listbox.pack(side="left", fill="y")

    scrollbar = tk.Scrollbar(top, orient="vertical")
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")

    listbox.config(yscrollcommand=scrollbar.set)

    for item in image_set:
        listbox.insert(tk.END, item)

    listbox.bind('<<ListboxSelect>>', on_select)

    top.mainloop()


if __name__=="__main__":
    # ============ Callback function to capture mouse events ============
    def handle_line_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, myAnn

        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_dragging()
            if myAnn.state.is_consecutive_line and myAnn.lines:
                # Start from the last point of the last line.
                points = [myAnn.lines[-1][1]]  # Last point of the last line
            else:
                points = [(x, y)]  # Reset points list with the new start

        elif event == cv2.EVENT_MOUSEMOVE and myAnn.state.is_dragging:
            temp_image = image.copy()
            cv2.line(temp_image, points[0], (x, y), myAnn.color, thickness=myAnn.thickness)
            cv2.imshow('image', temp_image)

        elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
            myAnn.state.end_draggging()
            points.append((x, y))  # Add end point
            myAnn.lines.append((points[0], points[1]))  # Store the line

            cv2.line(annotation, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            # Update temp_image with the final line for visual feedback
            cv2.line(image, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            cv2.imshow('image', image)  # Show the image with the final line
            ## Also clear redo history.
            myAnn.undone_lines = [] 

    # TODO: Scratch = drawing several lines in frame... (or maybe not using it?)
    def _handle_scratch_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, myAnn
        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_scratching()
            cv2.circle(annotation, (x, y), 1, myAnn.color, -1)

        elif event == cv2.EVENT_MOUSEMOVE and myAnn.state.is_scratching:
            # keep reading "points" and use cv2.line to connect them all 
            cv2.circle(annotation, (x, y), 1, myAnn.color, -1)
            cv2.imshow('image', cv2.addWeighted(image, 0.8, annotation, 0.5, 0))
            
        elif event == cv2.EVENT_LBUTTONUP:
            myAnn.state.end_scratching()

    def handle_nearest_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, myAnn

        roi_dim = 101  # (5x5px)
        detect_range = 8
        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_dragging()
            # Start from the nearest point.
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], detect_range)  

            # Same point: do nothing.
            if n_x==x and n_y==y:
                points = [(n_x, n_y)]
            else:
                temp_image = image.copy()
                # Show roi range.
                roi_top_left = (max(x - roi_dim // 2, 0), max(y - roi_dim // 2, 0))
                roi_bottom_right = (min(x + roi_dim // 2, image.shape[1]), min(y + roi_dim // 2, image.shape[0]))
                temp_image = add_semi_transparent_rectangle(temp_image, roi_top_left, roi_bottom_right, (0, 255, 0), 0.3)

                print(f"\r[INFO] Nearest point triggered: {n_x},{n_y}")
                cv2.line(temp_image, (n_x, n_y), (x, y), myAnn.color, thickness=myAnn.thickness)
                # Highlight the nearest point (n_x, n_y) with a red circle
                # # Note: n_x and n_y are reversed
                cv2.circle(temp_image, (n_x, n_y), radius=1, color=(0, 0, 255), thickness=-1)  # -1 fills the circle
                points = [(n_x, n_y)]

                cv2.imshow('image', temp_image)
                    

        elif event == cv2.EVENT_MOUSEMOVE and myAnn.state.is_dragging:
            temp_image = image.copy()
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], detect_range) 
            
            # Show roi range.
            roi_top_left = (max(x - roi_dim // 2, 0), max(y - roi_dim // 2, 0))
            roi_bottom_right = (min(x + roi_dim // 2, image.shape[1]), min(y + roi_dim // 2, image.shape[0]))
            temp_image = add_semi_transparent_rectangle(temp_image, roi_top_left, roi_bottom_right, (0, 255, 0), 0.3)

            if n_x==x and n_y==y: # same point
                # print(f"\r[INFO] Position: {x},{y}")
                cv2.line(temp_image, points[0], (x, y), myAnn.color, thickness=myAnn.thickness)
            else:
                print(f"\r[INFO] Nearest point triggered: coordinate: {n_x},{n_y}")
                cv2.line(temp_image, points[0], (n_x, n_y), myAnn.color, thickness=myAnn.thickness)
                # Highlight the nearest point (n_x, n_y) with a red circle
                for (_px, _py) in local_endpoints:
                    px, py = local2global((_px, _py), (x,y), roi_dim, image.shape[:2])
                    # highlight size: 2
                    cv2.circle(temp_image, (px, py), radius=2, color=(0, 0, 255), thickness=-1)  # -1 fills the circle
    
            cv2.imshow('image', temp_image)
            # print(f"Current pos: {n_x}, {n_y}")

        elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
            myAnn.state.end_draggging()
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], detect_range) 
            
            points.append((n_x, n_y))  # Add end point
            myAnn.lines.append((points[0], points[1]))  # Store the line

            cv2.line(annotation, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            # Update temp_image with the final line for visual feedback
            cv2.line(image, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            cv2.imshow('image', image)  # Show the image with the final line
            ## Also clear redo history.
            myAnn.undone_lines = [] 
            # print(f"Current pos: {n_x}, {n_y}")

    ## ========================= Main handler =========================
    def mouse_handler(event, x, y, flags, param):
        global points, image, temp_image, myAnn
        if myAnn.state.drawing_mode == "line":
            handle_line_mode(event, x, y, flags, param)
        elif myAnn.state.drawing_mode == "scratch":
            _handle_scratch_mode(event, x, y, flags, param)
        elif myAnn.state.drawing_mode == "nearest":
            handle_nearest_mode(event, x, y, flags, param)

    # Load your image
    #image_path = "/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/0727-0933-0272-img00.bmp"
    #annotation_path = "/home/pywu/Downloads/zhong/line_annotator/UV-only/0727-0933-0272-img00_UV.bmp"  # Change to your actual path
            
    ### Use argparse to load default value to those variables.
    args = parse_arguments()
    save_path = args.save_path
    demo_path = args.demo_path
    image_root = args.image_root
    annotation_root = args.annotation_root

    save_path_annotation = os.path.join(save_path, "annotation")
    save_path_origin = os.path.join(save_path, "image")
    save_path_combined = os.path.join(save_path, "combined")
    os.makedirs(demo_path, exist_ok=True)
    os.makedirs(save_path_annotation, exist_ok=True)
    os.makedirs(save_path_origin, exist_ok=True)
    os.makedirs(save_path_combined, exist_ok=True)
    
    # Load the whole dataset
    image_set, annotation_set = read_data_pairs(image_root, annotation_root)
    
    # Read the first image by selecting in tkinter.
    global current_image_index
    # current_image_index = 345
    current_image_index = select_image_annotation_pair_by_index(image_set, annotation_set)
    image_path = image_set[current_image_index]
    annotation_path = annotation_set[current_image_index]
    last_index = current_image_index

    # Initialize annotator
    image, annotation, temp_image, image_backup, annotation_backup, myAnn = initialize_annotator(image_path, annotation_path)
        
    # Create a window and bind the callback function to the window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_handler)
    cv2.imshow('image', image)

    while True:
        # Initialize image if the index changes. 
        print(current_image_index, last_index)
        if last_index != current_image_index:
            # Load and display the new image
            image_path = image_set[current_image_index]
            annotation_path = annotation_set[current_image_index]
            # TODO: the image will be changed after saving. IDK why.
            
            # # Read the first image by selecting in tkinter.
            # image_path, annotation_path = select_image_annotation_pair_by_index(image_set, annotation_set)
            
            # Initialize
            image, annotation, temp_image, image_backup, annotation_backup, myAnn = initialize_annotator(image_path, annotation_path)
            # Initial display with description
            last_index = current_image_index
            cv2.imshow('image', image)

        k = cv2.waitKey(0)

        # ====== Press 'n' to open the image selector======
        if k == ord('/'): 
            # threading.Thread(target=open_image_selector).start()
            # threading.Thread(target=lambda: open_image_selector(image_set), daemon=True).start()
            current_image_index = select_image_annotation_pair_by_index(image_set, annotation_set)

        # ====== Press 'x' to toggle drawing mode (not implemented.) ======
        elif k == ord('x'):
            # myAnn.state.drawing_mode = "scratch" if myAnn.state.drawing_mode == "line" else "line"
            # hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            hints = [f"Warning: Scratch mode not implemented."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn)            
        # # ====== [Default] Press 'n' to toggle nearest dragging mode ======
        elif k == ord('n'):
            myAnn.state.drawing_mode = "nearest"
            hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn) 
        # ====== Press 'l' to toggle line mode ======
        elif k == ord('l'):
            myAnn.state.drawing_mode = "line"
            hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn) 

        # ====== Press 'c' to toggle consecutive dragging mode ======
        elif k == ord('c'):
            if myAnn.state.drawing_mode == "line":
                myAnn.state.is_consecutive_line = not myAnn.state.is_consecutive_line
                hints = [f" {'Enable' if myAnn.state.is_consecutive_line else 'Disable'} consecutive dragging mode."]
                print_on_console(hints)
                print_on_image(hints, image, myAnn)
            else:
                hints = [f"Consecutive dragging is avaliable for drawing_mode only."]
                print_on_console(hints)

        # ====== Press 'u' to undo the last line drawn ======
        elif k == ord('u'):  
            if myAnn.lines:
                myAnn.undone_lines.append(myAnn.lines.pop())  # Move the last line to undone list
                image = image_backup.copy()  # Restore the previous state
                annotation = annotation_backup.copy()  # Restore the previous state
                for line in myAnn.lines:  # Redraw remaining lines
                    cv2.line(image, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                    cv2.line(annotation, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                cv2.imshow('image', image)
                print("[INFO] Undo.")
        # ====== Press 'r' to redo the last undone line ======
        elif k == ord('r'):  
            if myAnn.undone_lines:
                line = myAnn.undone_lines.pop()  # Get the last undone line
                myAnn.lines.append(line)  # Move it back to lines list
                cv2.line(image, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                cv2.line(annotation, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                cv2.imshow('image', image)
                print("[INFO] Redo.") 
        # ====== Press 'h' to show message ======
        elif k == ord('h'):  
            message = [
                "============= Welcome to Line Annotator! =============",
                "Press left click to draw a line.",
                "Undo: 'u'",
                "Redo: 'r'",
                "[Default]: Nearest mode: 'n'",
                "Line mode: 'l'",
                "Scratch mode: 'x'",
                "Save annotation: 's'",
                "Select another iamge: '/'",
                "Leave without Saveing: 'esc'",
                "=============== Author: Zhong-Wei Lin ===============", 
                "Show all endpoints: 'p'",
            ]
            print_on_console(message)
            print_on_image([os.path.basename(image_path)])

        # ====== Press 'p' to toggle to print all endpoints ======
        elif k == ord('p'):  
            temp_image = image.copy()
            endpoints = detect_endpoints_local(annotation, myAnn.color)
            message = [
                "Show all endpoints",
            ]
            print_on_console(message)
            print(endpoints)
            for (px, py) in endpoints:
                cv2.circle(temp_image, (px, py), radius=2, color=(0, 0, 255), thickness=-1)  # -1 fills the circle
            cv2.imshow('image', temp_image)

        # ====== Leave and Save ======
        elif k == ord('s'):  
            # Optional: Combine original and annotation images for visualization
            combined_image = cv2.addWeighted(image, 1, annotation, 1, 0)

            # Save demo
            cv2.imwrite(os.path.join(demo_path, 'original_image.jpg'), image_backup)
            cv2.imwrite(os.path.join(demo_path, 'annotation_only.jpg'), annotation)
            cv2.imwrite(os.path.join(demo_path, 'combined_image.jpg'), combined_image)

            # Save result
            image_name = os.path.basename(image_path)
            annotation_name = os.path.basename(annotation_path) 
            cv2.imwrite(os.path.join(save_path_origin, image_name), image_backup)
            cv2.imwrite(os.path.join(save_path_annotation, annotation_name), annotation)
            cv2.imwrite(os.path.join(save_path_combined, f"combined_{image_name}"), combined_image)

            message = [f"[INFO] Annotation is saved at {save_path}."]
            # print_on_image(message, image, myAnn, font_size=0.5)
            print_on_console(message)
            # break
        # ====== ESC key to leave without saving ======
        elif k == 27: 
            print("[INFO] Leave without saving.")
            break
        # ====== Exception handeling (show hint) ======
        else: 
            instruction_image = image.copy()
            message = [
                f"Current drawing mode: {myAnn.state.drawing_mode}",
                "Press 'h' for help",
            ]
            print_on_console(message)
            # print_on_image(message, image, myAnn, font_size=0.5)

    cv2.destroyAllWindows()

