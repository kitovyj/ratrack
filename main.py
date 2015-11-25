import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
from PIL import Image, ImageTk

from tracking import do_tracking, Animals, Animal, BodyPart
                        

frames = Queue.Queue()    
    
time_to_stop = threading.Event()

animals = Animals()

tracking_thread = threading.Thread(target = do_tracking, args = (frames, time_to_stop, animals))

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

adding_new_animal = False

class point:
    def __init__( self, x = 0, y = 0):
        self.x, self.y = x, y

new_animal_start = point()
new_animal_end = point()

def update_image():
    set_image(image_container, current_image)            
    
def draw_bodypart(bp):
    global current_image
    cv2.circle(current_image, (bp.center.x, bp.center.y), bp.radius, (255, 255, 255))
            
def draw_animals():
    for a in animals.animals:
        draw_bodypart(a.back)
        draw_bodypart(a.front)
        draw_bodypart(a.head)    
                
def draw_image():
    global current_image
    current_image = current_frame.copy()    
    draw_animals()
    if adding_new_animal:
        cv2.line(current_image, (new_animal_start.x, new_animal_start.y), (new_animal_end.x, new_animal_end.y), (255, 255, 255))                

def on_video_position_changed(val):
    global current_frame
    max = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    cap.set(cv2.CAP_PROP_POS_FRAMES, max * float(val) / max_video_position_slider_value)
    ret, frame = cap.read()
    current_frame = frame
    draw_image()
    update_image()
    
button = Tk.Button(root, text = "Quit", command = quit)
button.pack(side = Tk.LEFT)

slider = Tk.Scale(root, length = 300, from_ = 0, to = max_video_position_slider_value, 
                  orient = Tk.HORIZONTAL, command = on_video_position_changed)
                  
slider.pack(side = Tk.BOTTOM)
        
def on_left_mouse_button_down(event):  
    global adding_new_animal, new_animal_start
    new_animal_start.x = event.x
    new_animal_start.y = event.y
    adding_new_animal = True

def on_left_mouse_button_up(event):
    global adding_new_animal, new_animal_end, current_image, animals
    if adding_new_animal:
        new_animal_end.x = event.x
        new_animal_end.y = event.y
        adding_new_animal = False
        animals.add_animal(new_animal_start.x, new_animal_start.y, new_animal_end.x, new_animal_end.y)
        draw_image()
        update_image()            
    
    
def on_mouse_moved(event):
    global current_image, new_animal_end
    if adding_new_animal:
        new_animal_end.x = event.x
        new_animal_end.y = event.y
        draw_image()
        update_image()

image_container.bind('<Button-1>', on_left_mouse_button_down)
image_container.bind('<ButtonRelease-1>', on_left_mouse_button_up)
image_container.bind('<Motion>', on_mouse_moved)

# take first frame of the video
ret, current_frame = cap.read()
if not ret:
    print('can\'t read the video')
    sys.exit()

draw_image()
update_image()

def poll_queue():
    if not frames.empty():
        frame = frames.get()
        set_image(image_container, frame)
     
    
    root.after(100, poll_queue)

poll_queue()
root.mainloop()
root.destroy()

