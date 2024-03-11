import tkinter as tk
from tkinter import ttk

from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QWidget
import sys

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

def run_selector_app(image_list):
    app = QApplication(sys.argv)
    ex = ImageSelector(image_list)
    sys.exit(app.exec_())


def select_image_annotation_pair(image_set, annotation_set):
    root = tk.Tk()
    root.title('Select Image and Annotation')

    # Variable to store the index of the selected pair
    selected_index = tk.IntVar(value=-1)

    def on_select(event):
        selection_index = listbox.curselection()[0]  # Get selected index
        selected_index.set(selection_index)  # Set the index
        root.destroy()  # Close the window

    # Create listbox and populate it
    listbox = tk.Listbox(root, width=100, height=20)
    for idx, (image_path, annotation_path) in enumerate(zip(image_set, annotation_set)):
        listbox.insert(tk.END, f"{idx}: {image_path} - {annotation_path}")
    listbox.bind('<<ListboxSelect>>', on_select)
    listbox.pack()

    root.mainloop()  # Start the GUI event loop

    if selected_index.get() >= 0:  # If a selection was made
        return image_set[selected_index.get()], annotation_set[selected_index.get()]
    else:
        return None, None  # Or handle this case as you prefer
