import cv2
import numpy as np

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
