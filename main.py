import cv2
import numpy as np
import os
import time
from scipy.ndimage import convolve


class Annotator():
    def __init__(self) -> None:
        self.color = (255, 255, 255)  # White color
        self.thickness = 1  # Line thickness

        # List to store lines
        self.lines = []  # Store the lines drawn
        self.undone_lines = []  # Store the undone lines for redo functionality

        # State variable to store the flags
        self.state = AnnotatorState()


    def update_image(self, new_image):
        self.image = new_image

class AnnotatorState():
    def __init__(self) -> None:
        # Color, thickness for the annotation

        self.is_dragging = False # Flag to track if mouse was is_dragging to draw
        self.is_scratching = False # Flag to track if mouse was is_scratching
        self.leave_hint = False # Flag to record if hint info should disappear
        self.leave_help = False # Flag to record if help info should disappear
        self.drag_style = None # Add a flag for consecutive drawing mode.
        self.drawing_mode = "line"
        
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



def print_on_console(info_list: list):
    for line in info_list:
        print(line)

def print_on_image(info_list, image, myAnn, duration=2000, num_frame=25, font_size=0.7):
    # Show on cmd.
    temp_image = image.copy()
    myAnn.state.leave_hint = False
    y0, dy = 30, 30  # Initial position and line spacing
    for i, line in enumerate(info_list):
        y = y0 + i * dy
        cv2.putText(temp_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, myAnn.color, 2)
    cv2.imshow('image', temp_image)
    # Wait for 2 second
    for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
        if myAnn.state.leave_hint:
            break
        elif cv2.waitKey(duration//num_frame) != -1:  # Wait for 100 ms between frames
            myAnn.state.leave_hint=True
        
    # Fade out effect
    for alpha in np.linspace(1, 0, num=num_frame):  # Generate 10 steps from 1 to 0
        if myAnn.state.leave_hint:
            break
        faded_image = cv2.addWeighted(temp_image, alpha, image, 1 - alpha, 0)
        cv2.imshow('image', faded_image)
        if cv2.waitKey(1000//num_frame) != -1:  # Wait for 100 ms between frames
            myAnn.state.leave_hint=True
    cv2.imshow('image', image)


def extract_sub_image(image, x, y, roi_size):
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
    start_x = max(x - roi_size//2, 0)
    end_x = min(x + roi_size//2+1, width)  # +5 because upper bound is exclusive
    start_y = max(y - roi_size//2, 0)
    end_y = min(y + roi_size//2+1, height)
    
    # Extract the sub-image
    sub_image = image[start_y:end_y, start_x:end_x]
    
    return sub_image

# def local_to_global(local_coord, roi_center, image_size):
#     local_nx, local_ny = local_coord
#     x, y = roi_center
    
#     global_nx = x + local_nx
#     global_ny = y + local_ny
#     global_nx = max(0, min(global_nx, image_size[0] - 1))
#     global_ny = max(0, min(global_ny, image_size[1] - 1))

#     return global_nx, global_ny


def detect_endpoints(binary_map):
    # Define the kernel for 8-connectivity
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Convolve binary map with kernel to count neighbors
    neighbor_count = convolve(binary_map.astype(np.uint8), kernel, mode='constant', cval=0)
    
    # Endpoints are segment pixels with exactly one neighbor
    endpoints = (binary_map == 1) & (neighbor_count == 1)
    
    return np.transpose(np.nonzero(endpoints))
    # return np.nonzero(endpoints)


def find_nearest_point_on_map_within_radius(ROI, curser_pos, image_size, target_color, radius):
    """
    Find the nearest pixel position of target_color to the given position in the image.
    """
    
    #x, y = curser_pos # global

    h, w = ROI.shape[:2]
    assert h%2==1 and w%2==1
    cx = h//2
    cy = h//2
    # Create a boolean mask where the color matches the target color

    # mask = np.all(ROI == target_color, axis=-1)
    # # Find all y, x positions where mask is True
    # ones_x, ones_y = np.where(mask==True) # 
    # endpoints = np.transpose(np.array(find_segment_endpoints(ROI)))
    binary_map = np.where(np.all(ROI == target_color, axis=-1), 1, 0)
    endpoints = detect_endpoints(binary_map)
    ## TODO: find all endpoints in ROI.
    if len(endpoints) == 0:
        return curser_pos  # Return the original position if no target color found
    else:
        ones_x, ones_y = np.transpose(endpoints)
        # Calculate distances from center to all points with target color
        distances = (ones_x - cx) ** 2 + (ones_y - cy) ** 2
        # Find the index of the minimum distance
        if np.amin(distances) > radius**2:
            return curser_pos  # Return the original position
        else:
            nearest_index = np.argmin(distances)
            # Return the nearest point (note the reversal from y,x to x,y)
            local_nx = ones_x[nearest_index]
            local_ny = ones_y[nearest_index]
            ## Convert local coordinate in a 2D ROI to global coordinate value.
            gx, gy = curser_pos
            global_nx = local_nx - cx + gx
            global_ny = local_ny - cy + gy
            global_nx = max(0, min(global_nx, image_size[0] - 1))
            global_ny = max(0, min(global_ny, image_size[1] - 1))
        
    return (global_nx, global_ny)



# def _find_nearest_point_within_radius(annotation, position, radius, target_color):

#     endpoints = find_nearest_point_on_map(annotation, position, image.shape[:2], target_color)

#     if len(endpoints)==0:
#         print("near")
#         return position  # Return the original position if no points are provided

#     # Convert list of points to a numpy array for efficient computation
#     xy = np.array(endpoints)

#     # Calculate squared Euclidean distances from the given position to all points
#     d2 = np.sum((xy - np.array(position)) ** 2, axis=1)

#     # Check if there's any point within the radius
#     within_radius = d2 <= radius**2
#     if not np.any(within_radius):
#         return position  # Return the original position if no point is within the radius

#     # Find the index of the minimum distance within the radius
#     nearest_index = np.argmin(d2[within_radius])
#     # Get the actual nearest point within radius
#     nearest_point = xy[within_radius][nearest_index]

#     return tuple(nearest_point)


     
if __name__=="__main__":

    # ============ Callback function to capture mouse events ============
    def handle_line_mode(event, x, y, flags, param):
        global points, image, temp_image, annotation, myAnn

        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_dragging()
            if myAnn.state.drag_style=="consecutive" and myAnn.lines:
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

        roi_size = 50  # (5x5px)
        radius = 10
        ROI = extract_sub_image(annotation, x, y, roi_size)

        if event == cv2.EVENT_LBUTTONDOWN:
            myAnn.state.start_dragging()
            # Start from the nearest point.
            n_x, n_y = find_nearest_point_on_map_within_radius(ROI, (x, y), image.shape[:2], myAnn.color, radius)
            # n_x, n_y= find_nearest_point_within_radius(ROI, (x, y), radius, myAnn.color)       
            points = [(n_x, n_y)]
            # if n_x!=x or n_y!=y: # First draw
            #     myAnn.state.immediately_draw = True
            # print(f"Current pos: {n_x}, {n_y}")

        elif event == cv2.EVENT_MOUSEMOVE and myAnn.state.is_dragging:
            temp_image = image.copy()
            n_x, n_y = find_nearest_point_on_map_within_radius(ROI, (x, y), image.shape[:2], myAnn.color, radius)
            if n_x==x and n_y==y: # same point
                print("A")
                cv2.line(temp_image, points[0], (x, y), myAnn.color, thickness=myAnn.thickness)
            else:
                print("B")
                cv2.line(temp_image, points[0], (n_x, n_y), myAnn.color, thickness=myAnn.thickness)
            cv2.imshow('image', temp_image)
            # print(f"Current pos: {n_x}, {n_y}")

        elif event == cv2.EVENT_LBUTTONUP: # Record the drawing
            myAnn.state.end_draggging()
            n_x, n_y = find_nearest_point_on_map_within_radius(ROI, (x, y), image.shape[:2], myAnn.color, radius) 
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

    # Create Backups 
    temp_image = image.copy()  # Temporary image for showing the line preview
    image_backup = image.copy()  # Backup image for undo functionality
    annotation_backup = annotation.copy() # Backup image for undo functionality

    myAnn = Annotator()

    # Create a window and bind the callback function to the window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_handler)

    # Initial display with description
    cv2.imshow('image', image)

    while True:
        k = cv2.waitKey(0)
        # ====== Switch drawing mode (not implemented.) ======
        if k == ord('x'):
            # myAnn.state.drawing_mode = "scratch" if myAnn.state.drawing_mode == "line" else "line"
            # hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            hints = [f"Warning: Scratch mode not implemented."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn)            
        # # ====== Toggle nearest dragging mode ======
        elif k == ord('n'):
            myAnn.state.drawing_mode = "nearest"
            hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            # hints = [f"Warning: nearest mode not implemented."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn) 
            # if myAnn.state.drawing_mode == "nearest":
            #     myAnn.state.drag_style = "nearest"
            #     print("Nearest dragging mode enabled")
            # else:
            #     print(f"Current state: {myAnn.state.drawing_mode} is not suitable for nearest color line mode.")
        
        # ====== Toggle line mode ======
        elif k == ord('l'):
            myAnn.state.drawing_mode = "line"
            hints = [f"Switched to {myAnn.state.drawing_mode} mode."]
            print_on_console(hints)
            print_on_image(hints, image, myAnn) 

        # ====== Toggle consecutive dragging mode ======
        elif k == ord('c'):
            if myAnn.state.drawing_mode == "line":
                myAnn.state.drag_style = "consecutive"
                hints = [f" {'Enable' if myAnn.state.drag_style else 'Disable'} consecutive dragging mode."]
                print_on_console(hints)
                print_on_image(hints, image, myAnn)
            else:
                hints = [f"Consecutive dragging is avaliable for drawing_mode only."]
                print_on_console(hints)
        # ====== Toggle arbitrary dragging mode ======
        elif k == ord('a'):
            if myAnn.state.drawing_mode == "line":
                myAnn.state.drag_style = None
                hints = ["Arbitrary dragging mode."]
                print_on_console(hints)
                print_on_image(hints, image, myAnn)
            else:
                hints = [f"Arbitrary dragging is avaliable for drawing_mode only."]
                print_on_console(hints)

        # ====== Undo the last line drawn ======
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
        # ====== Redo the last undone line ======
        elif k == ord('r'):  
            if myAnn.undone_lines:
                line = myAnn.undone_lines.pop()  # Get the last undone line
                myAnn.lines.append(line)  # Move it back to lines list
                cv2.line(image, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                cv2.line(annotation, line[0], line[1], myAnn.color, thickness=myAnn.thickness)
                cv2.imshow('image', image)
                print("[INFO] Redo.") 
        # ====== Show/hide instructions ======
        # elif k == ord('h'):  
        #     instruction_image = image.copy()
        #     instructions = [
        #         "============= Welcome to Line Annotator! =============",
        #         "Press left click to draw a line.",
        #         "Undo: 'u'",
        #         "Redo: 'r'",
        #         "Scratch mode: 'x'",
        #         "Leave and Save: 's'",
        #         "Leave without Saveing: 'esc'",
        #         "(Under Line mode) Nearest: 'n';  Consecutive: 'c'; Arbitrary: 'a'",
        #         "=============== Author: Zhong-Wei Lin ===============", 
        #         "",
        #     ]
        #     print_on_console(instructions)
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
            # temp_image = image.copy()
            # # Display the description on the temporary image
            # hints = [
            #     "Hint:",
            #     "Press 'u' to undo, 'r' to redo.",
            #     "Press 's' to save and leave.",
            # ]
            # print_on_console(hints)
            # print_on_image(hints, image, myAnn)
            instruction_image = image.copy()
            instructions = [
                "============= Welcome to Line Annotator! =============",
                "Press left click to draw a line.",
                "Undo: 'u'",
                "Redo: 'r'",
                "Scratch mode: 'x'",
                "Leave and Save: 's'",
                "Leave without Saveing: 'esc'",
                "(Under Line mode) Nearest: 'n';  Consecutive: 'c'; Arbitrary: 'a'",
                "=============== Author: Zhong-Wei Lin ===============", 
                "",
            ]
            print_on_console(instructions)
            print_on_image(instructions, image, myAnn, font_size=0.5)

    cv2.destroyAllWindows()

