import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QWidget
import sys, os
import glob

class ImageSelector(QWidget):
    def __init__(self, image_list):
        super().__init__()
        self.initUI(image_list)

    def initUI(self, image_list):
        layout = QVBoxLayout(self)
        self.listWidget = QListWidget()
        self.listWidget.addItems(image_list)
        self.listWidget.clicked.connect(self.listClicked)
        layout.addWidget(self.listWidget)
        self.setLayout(layout)
        self.setWindowTitle('Select an Image')
        self.setGeometry(100, 100, 400, 600)
        self.show()

    def listClicked(self, qModelIndex):
        global current_image_index
        current_image_index = qModelIndex.row()
        self.close()


# ======================== Call image selector ========================
def run_selector_app(image_list):
    app = QApplication(sys.argv)
    ex = ImageSelector(image_list)
    sys.exit(app.exec_())

# ======================== Read Image by given path ========================
def read_image_and_annotation(image_path, annotation_path):

    image = cv2.imread(image_path)

    # Load previous annotations
    if os.path.exists(annotation_path):
        print(f"[INFO] Load existing annotation from {annotation_path}")
        annotation = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        if annotation.ndim == 2 or annotation.shape[2] == 1:  # If the loaded annotation is grayscale
            annotation = cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)
    else:
        # Create a blank image (black image) for drawing annotations
        annotation = np.zeros_like(image)
     
    return image, annotation

# ======================== Read image by Tkinter ========================
def select_image_annotation_pair_by_index(image_set, 
                                          annotation_set, 
                                          window_size_ratio=(0.3,0.8), 
                                          font=('Helvetica', 30, 'normal')):
    
    # from tkinter import font
    # root = tk.Tk()
    # fonts_list = font.families()
    # root.destroy()

    # for font in fonts_list:
    #     print(font)
        
    root = tk.Tk()
    root.title('Select Image')

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate window size as a percentage of screen size (e.g., 50% of the screen)
    pw, ph = window_size_ratio
    window_width = int(screen_width * pw)
    window_height = int(screen_height * ph)

    # Optionally, calculate the position of window
    x_position = 0
    y_position = 0

    # Set window size and position
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Variable to store the index of the selected pair
    selected_index = tk.IntVar(value=-1)

    # Create a Frame to hold the Listbox and Scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create the vertical scrollbar
    scrollbar = tk.Scrollbar(frame, orient="vertical", width=window_width*0.05)

    # Create the Listbox and configure it to use the Scrollbar
    listbox = tk.Listbox(frame, width=400, height=window_height, font=font, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    # Pack the Scrollbar to the right side of the Frame, and make it fill the Y-axis
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    # Pack the Listbox to fill the rest of the Frame
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    for idx, (image_path, _) in enumerate(zip(image_set, annotation_set)):
        listbox.insert(tk.END, f"{idx}: {os.path.basename(image_path)}")

    # Set callback functions
    def on_mouse_down(event):
        try:
            index = listbox.nearest(event.y)
            listbox.selection_set(index)
        except Exception as e:
            print("Error on mouse down:", e)

    def on_mouse_up(event):
        try:
            index = listbox.curselection()[0]
            selected_index.set(index)
            root.destroy()
        except Exception as e:
            print("Error on mouse up:", e)

    listbox.bind('<ButtonPress-1>', on_mouse_down)
    listbox.bind('<ButtonRelease-1>', on_mouse_up)

    root.mainloop()  # Start the GUI event loop

    if selected_index.get() >= 0:  # If a selection was made
        return selected_index.get() 
    else:
        return None  # Or handle this case as you prefer
    

def select_existing_annotation(result_path, window_size_ratio=(0.3,0.8), font=('Helvetica', 30, 'normal')):
    root = tk.Tk()
    root.title('Select Existing Annotation')

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate window size as a percentage of screen size (e.g., 50% of the screen)
    pw, ph = window_size_ratio
    window_width = int(screen_width * pw)
    window_height = int(screen_height * ph)

    # Optionally, calculate the position of window
    x_position = 0
    y_position = 0

    # Set window size and position
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Variable to store the string of selected path
    selected_path = tk.StringVar(value="")    

    # Create a Frame to hold the Listbox and Scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create the vertical scrollbar
    scrollbar = tk.Scrollbar(frame, orient="vertical", width=window_width*0.05)

    # Create the Listbox and configure it to use the Scrollbar
    listbox = tk.Listbox(frame, width=400, height=window_height, font=font, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    # Pack the Scrollbar to the right side of the Frame, and make it fill the Y-axis
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    # Pack the Listbox to fill the rest of the Frame
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create listbox and populate it
    annotation_files = sorted(glob.glob(os.path.join(result_path, "*.bmp")))  # Adjust pattern if necessary
    if not annotation_files:  # No annotation files found
        print("No annotations found in", result_path)
        return None

    for idx, file_path in enumerate(annotation_files):
        annotation_name = os.path.basename(file_path)
        listbox.insert(tk.END, f"{idx}: {annotation_name}")

    # Set callback functions
    def on_mouse_down(event):
        try:
            index = listbox.nearest(event.y)  # Get the item nearest to the mouse click
            listbox.selection_set(index)  # Highlight the item
        except Exception as e:
            print("Error on mouse down:", e)

    def on_mouse_up(event):
        try:
            if listbox.curselection():  # Check if there's a selection
                selection_index = listbox.curselection()[0]
                selected_path.set(annotation_files[selection_index])
            root.destroy()
        except Exception as e:
            print("Error on mouse up:", e)

    listbox.bind('<ButtonPress-1>', on_mouse_down)
    listbox.bind('<ButtonRelease-1>', on_mouse_up)
    
    root.mainloop()

    return selected_path.get()
