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
def select_image_annotation_pair_by_index(image_set, annotation_set, window_size_ratio=(0.3,0.8), font=('Helvetica', 30, 'normal')):
    
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

    def on_select(event):
        selection_index = listbox.curselection()[0]  # Get selected index
        selected_index.set(selection_index)  # Set the index
        root.destroy()  # Close the window

    # Create listbox and populate it
    listbox = tk.Listbox(root, width=400, height=window_height, font=font)
    for idx, (image_path, annotation_path) in enumerate(zip(image_set, annotation_set)):
        img_name = os.path.basename(image_path)
        listbox.insert(tk.END, f"{idx}: {img_name}")
    listbox.bind('<<ListboxSelect>>', on_select)
    listbox.pack()

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

    selected_path = tk.StringVar(value="")

    annotation_files = sorted(glob.glob(os.path.join(result_path, "*.bmp")))  # Adjust pattern if necessary
    if not annotation_files:  # No annotation files found
        print("No annotations found in", result_path)
        return None

    def on_select(event):
        selection_index = listbox.curselection()[0]
        selected_path.set(annotation_files[selection_index])
        root.destroy()

    listbox = tk.Listbox(root, width=window_width, height=window_height, font=font)
    for idx, file_path in enumerate(annotation_files):
        annotation_name = os.path.basename(file_path)
        listbox.insert(tk.END, f"{idx}: {annotation_name}")

    listbox.bind('<<ListboxSelect>>', on_select)
    listbox.pack()

    root.mainloop()

    return selected_path.get()
