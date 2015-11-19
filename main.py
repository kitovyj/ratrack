import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
from PIL import Image, ImageTk

from tracking import do_tracking
                        

frames = Queue.Queue()    
    
time_to_stop = threading.Event()

tracking_thread = threading.Thread(target = do_tracking, args = (frames, time_to_stop))

# tracking_thread.start()

root = Tk.Tk()

w = Tk.Label(root, text="Hello, world!")
w.pack()

def quit():
    time_to_stop.set()    
    tracking_thread.join()
    root.quit()
    
def set_image(container, matrix):
    img = Image.fromarray(matrix)
    imgtk = ImageTk.PhotoImage(image = img) 
    container.configure(image = imgtk)
    container.image = imgtk        

cap = cv2.VideoCapture('videotest.avi')
cap.set(cv2.CAP_PROP_POS_FRAMES, 190)

image_container = Tk.Label(root)
image_container.pack()

max_video_position_slider_value = 100

def on_video_position_changed(val):
    max = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    cap.set(cv2.CAP_PROP_POS_FRAMES, max * float(val) / max_video_position_slider_value)
    ret, frame = cap.read()
    set_image(image_container, frame)
            
    
button = Tk.Button(root, text = "Quit", command = quit)
button.pack(side = Tk.LEFT)

slider = Tk.Scale(root, length = 300, from_ = 0, to = max_video_position_slider_value, 
                  orient = Tk.HORIZONTAL, command = on_video_position_changed)
                  
slider.pack(side = Tk.BOTTOM)

start_x = 0
start_y = 0
end_x = 0
end_y = 0

def on_left_mouse_button_down(event):
    start_x = event.x
    start_y = event.y

def on_left_mouse_button_up(event):
    end_x = event.x
    end_y = event.y
    
def on_mouse_moved(event):
    return 0

image_container.bind('<Button-1>', on_left_mouse_button_down)
image_container.bind('<ButtonRelease-1>', on_left_mouse_button_up)
image_container.bind('<Motion>', on_mouse_moved)

# take first frame of the video
ret, frame = cap.read()
if not ret:
    print('can\'t read the video')
    sys.exit()

set_image(image_container, frame)

def poll_queue():
    if not frames.empty():
        frame = frames.get()
        set_image(image_container, frame)
     
    
    root.after(100, poll_queue)

poll_queue()
root.mainloop()
root.destroy()

