import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QWidget
import sys, os

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
        # Plot previous annotations on image. 
        image = cv2.addWeighted(image, 1, annotation, 0.5, 0)
    else:
        # Create a blank image (black image) for drawing annotations
        annotation = np.zeros_like(image)
     
    return image, annotation

# ======================== Read image by Tkinter ========================
def select_image_annotation_pair_by_index(image_set, annotation_set):
    root = tk.Tk()
    root.title('Select Image')

    # Variable to store the index of the selected pair
    selected_index = tk.IntVar(value=-1)

    def on_select(event):
        selection_index = listbox.curselection()[0]  # Get selected index
        selected_index.set(selection_index)  # Set the index
        root.destroy()  # Close the window

    # Create listbox and populate it
    listbox = tk.Listbox(root, width=100, height=20)
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
