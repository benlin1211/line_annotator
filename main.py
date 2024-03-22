import cv2
import numpy as np
import os
import time
from scipy.ndimage import convolve
import glob, re
# import tkinter as tk
# from tkinter import ttk
from utils.data_selector import select_image_annotation_pair_by_index, \
                                read_image_and_annotation, \
                                select_existing_annotation #, run_selector_app
from utils.printer import print_on_console, background_toggler #, print_on_image
import threading

import tkinter as tk
from tkinter import simpledialog
import argparse


class Action():
    def __init__(self, action_type: str, details: dict):
        self.action_type = action_type  # 'line' or 'erase'
        self.details = details 
        # line: line
        # erase: center, erase_radius

class Annotator():
    def __init__(self) -> None:
        self.color = (255, 255, 255)  # White color
        self.thickness = 1  # Line thickness
        self.show_background = True

        # List to store lines
        self.actions = []  # Store the actions 
        self.undone_actions = []  # Store the undone actions for redo functionality
        # State variable to store the flags
        self.state = AnnotatorState()

    ## TODO: maybe write the undo/redo function here?

class AnnotatorState():
    def __init__(self) -> None:
        # Color, thickness for the annotation

        self.is_dragging = False # Flag to track if mouse was is_dragging to draw
        self.leave_hint = False # Flag to record if hint info should disappear
        self.leave_help = False # Flag to record if help info should disappear
        self.is_consecutive_line = None # Add a flag for consecutive drawing mode.
        self.drawing_mode = "draw" # ["draw", "eraser"]
        
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
        self.leave_hint = False 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image and Annotation Loader with Custom Save Path.')
    parser.add_argument('--save_path', type=str, default='./result/', help='Path to save the output results.')
    parser.add_argument('--demo_path', type=str, default='./demo/', help='Path to save the demo.')
    parser.add_argument('--image_root', type=str, default='/home/pywu/Downloads/zhong/dataset/teeth_qisda/imgs/0727-0933/', help='Root directory for images.')
    parser.add_argument('--annotation_root', type=str, default='/home/pywu/Downloads/zhong/dataset/teeth_qisda/supplements/0727-0933/UV-only/', help='Root directory for annotations.')
        
    # Add stride and roi_dim arguments
    parser.add_argument('--stride_draw', type=int, default=10, help='Stride size for draw mode adjustments.')
    parser.add_argument('--stride_erase', type=int, default=10, help='Stride size for erase mode adjustments.')
    parser.add_argument('--roi_dim', type=int, default=201, help='ROI size for sub-image extraction.')
    
    args = parser.parse_args()
    return args


def read_data_pairs(image_set, annotation_set):
    annotation_set = sorted(glob.glob(os.path.join(annotation_root, "*.bmp")))
    sequence_numbers = []

    # Read annotation first.
    for a in annotation_set:
        number = os.path.basename(a).split("-")[-2]
        sequence_numbers.append(number)

    pattern = re.compile(r"-img0[0-2]\.bmp$")
     # Read image referring to annotation.
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
    # I relaxed to condition to scan single points as endpoints, too
    endpoints = (binary_map == 1) & (neighbor_count <= 1)

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

    image = cv2.imread(image_path)
    annotation = np.zeros_like(image)
    
    # Load previous annotations
    if os.path.exists(annotation_path):
        print(f"[INFO] Load existing annotation from {annotation_path}")
        annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        if annotation.ndim == 2 or annotation.shape[2] == 1:  # If the loaded annotation is grayscale
            annotation = cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)
    
    # Create Backups 
    temp_image = image.copy()  # Temporary image for showing the line preview
    image_backup = image.copy() # Backup image for undo functionality
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

    def handle_eraser_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, annotation_backup, myAnn
        # if event == cv2.EVENT_MOUSEMOVE:
        #     temp_image = image.copy()
        #     # Show stride range
        #     overlay = temp_image.copy()
        #     # Draw a circle
        #     cv2.circle(overlay, (x, y), stride, (0, 255, 255), -1)  # Drawing the circle on the overlay
        #     # Alpha value controls the transparency level (between 0 and 1)
        #     alpha = 0.4
        #     cv2.addWeighted(overlay, alpha, temp_image, 1-alpha, 0, temp_image)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Erasing by drawing a circle of the background color on annotation
            # Adjust the radius to change the eraser size
            # action = Action('erase', { "center": (x, y), "erase_radius": eraser_radius)
            # myAnn.actions.append(action)  # Store the line as an action

            cv2.circle(annotation, (x, y), radius=stride_erase, color=(0, 0, 0), thickness=-1)
            # Redraw the whole image
            if myAnn.show_background:
                image = cv2.addWeighted(image_backup, 1, annotation, 1, 0)
            else: 
                image = annotation
            cv2.imshow('image', image)

            ## Once the erase mode is used, ALL action stacks will be cleaned up.
            # TODO: I can't come up with a better idea...
            # Update annotation 
            annotation_backup = annotation
            # Also clear ALL history.
            myAnn.undone_actions = [] 
            myAnn.actions = [] 


    def handle_nearest_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, myAnn

        # roi_dim = 101  # (5x5px)
        # stride = 10
        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_dragging()
            # Start from the nearest point.
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], stride_draw)  

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

                # Highlight the nearest point (n_x, n_y) with a green circle
                # # Note: n_x and n_y are reversed
                cv2.circle(temp_image, (n_x, n_y), radius=1, color=(0, 255, 0), thickness=-1)  # -1 fills the circle
                points = [(n_x, n_y)]

                cv2.imshow('image', temp_image)
                

        elif event == cv2.EVENT_MOUSEMOVE and myAnn.state.is_dragging:
            temp_image = image.copy()
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], stride_draw) 
            
            # Show roi range.
            roi_top_left = (max(x - roi_dim // 2, 0), max(y - roi_dim // 2, 0))
            roi_bottom_right = (min(x + roi_dim // 2, image.shape[1]), min(y + roi_dim // 2, image.shape[0]))
            temp_image = add_semi_transparent_rectangle(temp_image, roi_top_left, roi_bottom_right, (0, 255, 0), 0.3)

            # Show stride range
            overlay = temp_image.copy()
            # Draw a circle
            cv2.circle(overlay, (x, y), stride_draw, (0, 255, 255), -1)  # Drawing the circle on the overlay
            # Alpha value controls the transparency level (between 0 and 1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, temp_image, 1-alpha, 0, temp_image)

            # Show all endpoints in rectangle with red color when dragging
            for (_px, _py) in local_endpoints:
                px, py = local2global((_px, _py), (x,y), roi_dim, image.shape[:2])
                # highlight size: 2
                cv2.circle(temp_image, (px, py), radius=2, color=(0, 0, 255), thickness=-1)  # -1 fills the circle
                
            # Point determinator
            if n_x==x and n_y==y: # same point
                # print(f"\r[INFO] Position: {x},{y}")
                cv2.line(temp_image, points[0], (x, y), myAnn.color, thickness=myAnn.thickness)
            else:
                print(f"\r[INFO] Nearest point triggered: coordinate: {n_x},{n_y}")
                cv2.line(temp_image, points[0], (n_x, n_y), myAnn.color, thickness=myAnn.thickness)
                # Highlight the nearest point pair (n_x, n_y) with a green circle
                cv2.circle(temp_image, points[0], radius=2, color=(0, 255, 0), thickness=-1)  # -1 fills the circle
                cv2.circle(temp_image, (n_x, n_y), radius=2, color=(0, 255, 0), thickness=-1)  # -1 fills the circle

    
            cv2.imshow('image', temp_image)
            # print(f"Current pos: {n_x}, {n_y}")

        elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
            myAnn.state.end_draggging()
            ROI = extract_sub_image(annotation, x, y, roi_dim)
            local_endpoints = detect_endpoints_local(ROI, myAnn.color)
            n_x, n_y = find_nearest_point_on_map_within_range(roi_dim, local_endpoints, (x, y), image.shape[:2], stride_draw) 
            
            points.append((n_x, n_y))  # Add end point
            action = Action('line', {'line':(points[0], points[1])})
            myAnn.actions.append(action)  # Store the line as an action

            cv2.line(annotation, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            # Update temp_image with the final line for visual feedback
            cv2.line(image, points[0], points[1], myAnn.color, thickness=myAnn.thickness)
            cv2.imshow('image', image)  # Show the image with the final line
            ## Also clear redo history.
            myAnn.undone_actions = [] 

    ## ===================== Main handler ============================
    def mouse_handler(event, x, y, flags, param):
        global points, image, temp_image, myAnn
        if myAnn.state.drawing_mode == "draw":
            handle_nearest_mode(event, x, y, flags, param)
        elif myAnn.state.drawing_mode == "eraser":
            handle_eraser_mode(event, x, y, flags, param)
    ## ===================== End of Main handler =====================


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
    ## image: 正在畫的圖
    ## annotation: 正在畫的標註
    ## image_backup: 原圖
    ## annotation_backup: 標註紀錄（for undo）

    # Plot previous annotations on image. 
    image = cv2.addWeighted(image, 1, annotation, 1, 0)

    # Show image with annotation, or annotation only.
    image = background_toggler(image_backup, annotation, myAnn.show_background)

    # Create a window and bind the callback function to the window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_handler)
    cv2.imshow('image', image)

    # Use global to make callback function sees the variable
    global stride_draw, roi_dim, stride_erase
    stride_draw = args.stride_draw
    stride_erase = args.stride_erase
    roi_dim = args.roi_dim

    # ================================== Main program ==================================
    while True:
        # Initialize image if the index changes. 
        print(f"Current_image_index={current_image_index}")
        print(f"image_name: {os.path.basename(image_path)}")
        if last_index != current_image_index:
            # Load and display the new image
            image_path = image_set[current_image_index]
            annotation_path = annotation_set[current_image_index]
            # TODO: the image will be changed after saving. IDK why.
            
            # # Read the first image by selecting in tkinter.
            # image_path, annotation_path = select_image_annotation_pair_by_index(image_set, annotation_set)
            
            # Initialize
            image, annotation, temp_image, image_backup, annotation_backup, myAnn = initialize_annotator(image_path, annotation_path)
            ## image: 正在畫的圖
            ## annotation: 正在畫的標註
            ## image_backup: 原圖
            ## annotation_backup: 標註紀錄（for undo）

            # Plot previous annotations on image. 
            image = cv2.addWeighted(image, 1, annotation, 1, 0)

            # Initial display with description
            last_index = current_image_index
            cv2.imshow('image', image)

        k = cv2.waitKey(0)
        # ====== Press 'n' to open the image selector======
        if k == ord('/'): 
            # threading.Thread(target=open_image_selector).start()
            # threading.Thread(target=lambda: open_image_selector(image_set), daemon=True).start()
            current_image_index = select_image_annotation_pair_by_index(image_set, annotation_set)
        # ====== Press 'e' to toggle erase mode ====== 
        elif k == ord('o'):
            myAnn.state.drawing_mode = "eraser"
            hints = [f"Switched to limited eraser mode."]
            print_on_console(hints)
            # print_on_image(hints, image, myAnn) 
        # # ====== [Default] Press 'n' to toggle nearest dragging mode ======
        elif k == ord('i'):
            myAnn.state.drawing_mode = "draw"
            hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            print_on_console(hints)
            # print_on_image(hints, image, myAnn) 
        # ====== Press 'c' to toggle consecutive dragging mode ======
        elif k == ord('c'):
            if myAnn.state.drawing_mode == "line":
                myAnn.state.is_consecutive_line = not myAnn.state.is_consecutive_line
                hints = [f" {'Enable' if myAnn.state.is_consecutive_line else 'Disable'} consecutive dragging mode."]
                print_on_console(hints)
                # print_on_image(hints, image, myAnn)
            else:
                hints = [f"Consecutive dragging is avaliable for drawing_mode only."]
                print_on_console(hints)

        # ====== Press 'u' to undo the last line drawn ======
        elif k == ord('u'):  
            print(len( myAnn.actions))
            if myAnn.actions:
                # Move the last line to undone list
                myAnn.undone_actions.append(myAnn.actions.pop())  
                # init 
                image = image_backup.copy() if myAnn.show_background else annotation_backup.copy()  # Restore the previous state
                annotation = annotation_backup.copy() 
                # Redo the remaining actions.
                for action in myAnn.actions: 
                    print(action.action_type)
                    if action.action_type == "line":
                        line = action.details["line"]
                        
                        # Redraw recorded lines
                        cv2.line(annotation, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                        ## Redraw remaining lines
                        # cv2.line(image, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                    # elif action.action_type == "erase":
                    #     x, y = action.details["center"]
                    #     erase_radius = action.details["erase_radius"]
                    #     cv2.circle(annotation, (x, y), radius=erase_radius, color=(0, 0, 0), thickness=-1)

                # Redraw the annotation result on image.
                image = cv2.addWeighted(image, 1, annotation, 1, 0)
                cv2.imshow('image', image)
                print("[INFO] Undo.")
        # ====== Press 'r' or right arrow key to redo the last undone line ======
        elif k == ord('r'):  
            if myAnn.undone_actions:
                action = myAnn.undone_actions.pop()  # Get the last undone line
                myAnn.actions.append(action)  # Move it back to lines list
                for action in myAnn.actions: 
                    print(action.action_type)
                if action.action_type == "line":
                    line = action.details["line"]
                    cv2.line(annotation, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                    # Redraw the annotation result on image.
                    image = cv2.addWeighted(image, 1, annotation, 1, 0)
                # elif action.action_type == "erase":
                #     image = cv2.addWeighted(image_backup, 1, annotation, 1, 0)

                
                cv2.imshow('image', image)
                print("[INFO] Redo.") 
        # ====== Press 'h' to show message ======
        elif k == ord('h'):  
            message = [
                "============= Welcome to Line Annotator! =============",
                "Press left click to draw a line.",
                "[Default]: Draw mode: 'i'",
                "Eraser mode: 'o'",
                "=============== Load and Save ===============", 
                "Save annotation: 's'",
                "Load another iamge: '/'",
                "Load previous annotations: '.'",
                "Leave without Saveing: 'esc'",
                "=============== Tools ===============", 
                "Undo: 'u' ",
                "Redo: 'r' ",
                "Show all endpoints: 'p'",
                "Decrease stride: 'left arrow key'",
                "Increase stride: 'right arrow key'",
            ]
            print_on_console(message)
            # print_on_image([os.path.basename(image_path)])

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
            combined_image = cv2.addWeighted(image_backup, 1, annotation, 1, 0)

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

        # ====== Decrease stride size with left arrow key ======
        elif k == 81:  
            if myAnn.state.drawing_mode == "draw":
                stride_draw = max(1, stride_draw - 1)
                message = [f"[INFO] Drawing stride decreased to {stride_draw}."]
            elif myAnn.state.drawing_mode == "eraser":
                stride_erase = max(1, stride_erase - 1)
                message = [f"[INFO] Drawing stride decreased to {stride_erase}."] 
            print_on_console(message)
        
        # ====== Increase stride size with right arrow key ======
        elif k == 83: 
            if myAnn.state.drawing_mode == "draw":
                max_stride_draw = roi_dim // 2  # Calculate the max stride based on roi_dim
                stride_draw = min(max_stride_draw, stride_draw + 1)
                message = [f"[INFO] Drawing stride increased to {stride_draw}."]
            if myAnn.state.drawing_mode == "eraser":
                max_stride_erase = roi_dim // 2  # Calculate the max stride based on roi_dim
                stride_erase = min(max_stride_erase, stride_erase + 1)
                message = [f"[INFO] Drawing stride increased to {stride_erase}."]
            print_on_console(message)     

        # ====== Toggle background ======
        elif k == ord('b'):
            myAnn.show_background = not myAnn.show_background 
            image = background_toggler(image_backup, annotation, myAnn.show_background)
            cv2.imshow('image', image)
        # ====== Load existinge annotation from save_path_annotation ======
        elif k == ord('.'):
            new_annotation_path = select_existing_annotation(save_path_annotation) # or use args.save_path if you want it configurable
            if new_annotation_path:
                # Update the display or any internal state as necessary
                image, annotation = read_image_and_annotation(image_path, new_annotation_path)
                # Plot previous annotations on image. 
                image = cv2.addWeighted(image, 1, annotation, 1, 0)
                print(f"Loaded annotation from {new_annotation_path}")
                cv2.imshow('image', image)
            else:
                print("Annotation selection was canceled or no annotations available.")
        
        # ====== Exception handeling (show hint) ======
        else: 
            instruction_image = image.copy()
            message = [
                f"Current drawing mode: {myAnn.state.drawing_mode}",
                "Press 'h' for help",
            ]
            print_on_console(message)
            # print_on_image(message, image, myAnn, font_size=0.5)

    # ================================== End main program ==================================
    cv2.destroyAllWindows()

