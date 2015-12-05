import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
import tkFileDialog

from PIL import Image, ImageTk

from tracking import Tracking, Animals, Animal, BodyPart, TrackingFlowElement

# tkinter layout management : http://zetcode.com/gui/tkinter/layout/                        

class point:
    def __init__( self, x = 0, y = 0):
        self.x, self.y = x, y
        
def calculate_scale_factor(src_width, src_height, dst_width, dst_height):
    k = float(src_width) / src_height
    if k > float(dst_width) / dst_height:
        f = float(dst_width) / src_width
        return (f, 0, (dst_height - dst_width / k) / 2)
    else:
        f = float(dst_height) / src_height        
        return (f, (dst_width - dst_height * k) / 2, 0)

def fit_image(image, width, height):
    
    rows, cols = image.shape[:2]
        
    k = float(cols) / rows
                
    if k > float(width) / height:
        cols = width
        rows = cols / k
    else:
        rows = height
        cols = rows * k
        
    return cv2.resize(image, (int(cols), int(rows)))

# silly python code to set button size in pixels
def create_button(root, ptext, pcommand, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    button = Tk.Button(f, text = ptext, command = pcommand)    
    button.pack(fill = Tk.BOTH, expand = 1)
    return button

class Gui:
    
    tracking_flow = Queue.Queue()    
    
    time_to_stop = threading.Event()

    current_frame_number = 0

    tracking_thread = 0
    
    video_file_name = 'videotest.avi'
    
    adding_new_animal = False
    new_animal_start = point()
    new_animal_end = point()
    
    image_width = 500
    image_height = 400

#    filtered_image_width = 320
    filtered_image_width = 470
    filtered_image_height = 370
    
    image_scale_factor = 0
    image_dx = 0
    image_dy = 0
        
    def __init__(self):
    
        self.root = Tk.Tk()
        self.root.geometry('1200x460')
    
        self.image_container = Tk.Label(self.root)
        self.image_container.place(x = 180, y = 20)
        # have to set fake image to switch 'width' and 'height' interpretation mode
        self.image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
        self.image_container.config(image = self.image_container.image)
        self.image_container.config(relief = Tk.GROOVE, width = self.image_width, height = self.image_height)
#        self.image_container.config(borderwidth = 1)
        self.image_container.bind('<Button-1>', self.on_left_mouse_button_down)
        self.image_container.bind('<ButtonRelease-1>', self.on_left_mouse_button_up)
        self.image_container.bind('<Motion>', self.on_mouse_moved)

        self.filtered_image_container = Tk.Label(self.root, text = 'test')
        self.filtered_image_container.place(x = 700, y = 20)
        self.filtered_image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
        self.filtered_image_container.config(image = self.filtered_image_container.image)
        self.filtered_image_container.config(relief = Tk.GROOVE, width = self.filtered_image_width, height = self.filtered_image_height)

        self.select_file_button = create_button(self.root, "Select file", self.select_file, 8, 25, 160, 30)
        self.start_button = create_button(self.root, "Start", self.start, 8, 60, 160, 30)
        self.quit_button = create_button(self.root, "Quit", self.quit, 8, 100, 160, 30)

        self.on_new_video()
        
        self.max_video_position_slider_value = 100
        self.slider = Tk.Scale(self.root, length = 500, from_ = 0, to = self.max_video_position_slider_value, 
                  orient = Tk.HORIZONTAL, command = self.on_video_position_changed)                  
        self.slider.place(x = 180, y = 400)
        
    def on_new_video(self):
        self.video = cv2.VideoCapture(self.video_file_name)
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, 190)        

        # take first frame of the video
        ret, self.current_frame = self.video.read()

        if not ret:
            print('can\'t read the video')
            sys.exit()
            
        rows, cols = self.current_frame.shape[:2]
        
        self.tracking = Tracking(cols, rows)     
        
        (self.image_scale_factor, self.image_dx, self.image_dy) = calculate_scale_factor(cols, rows, self.image_width, self.image_height)
        
        self.draw_image()
        self.update_image()
        
    def run(self):
        
        self.root.mainloop()
        
        self.root.destroy()
        self.video.release() 
            
    def set_image(self, container, matrix):
        img = Image.fromarray(matrix)
        imgtk = ImageTk.PhotoImage(image = img) 
        container.image = imgtk        
        container.configure(image = container.image)

    def update_image(self):
        self.set_image(self.image_container, self.current_image)            
    
    def draw_bodypart(self, bp):
        c = bp.get_center()
        c.x = c.x * self.image_scale_factor
        c.y = c.y * self.image_scale_factor
        r = bp.get_radius()
        cv2.circle(self.current_image, (int(c.x), int(c.y)), 
                   int(r * self.image_scale_factor), (255, 255, 255))
            
    def draw_animals(self):
        for a in self.tracking.animals.animals:
            self.draw_bodypart(a.back)
            self.draw_bodypart(a.front)
            self.draw_bodypart(a.head)    
                
    def draw_image(self):
        
        self.current_image = self.current_frame.copy()
        self.current_image = fit_image(self.current_image, self.image_width, self.image_height)
        
        self.draw_animals()
        if self.adding_new_animal:
            cv2.line(self.current_image, (int(self.new_animal_start.x), int(self.new_animal_start.y)), 
                     (int(self.new_animal_end.x), int(self.new_animal_end.y)), (255, 255, 255))                
    
    def poll_tracking_flow(self):
            
        if not self.tracking_flow.empty():
            e = self.tracking_flow.get()
            ret, self.current_frame = self.video.read()
            self.draw_image()
            self.update_image()         
            self.set_image(self.filtered_image_container, 
                           fit_image(e.filtered_image, self.filtered_image_width, self.filtered_image_height))            
            
        self.root.after(100, self.poll_tracking_flow)

    def on_video_position_changed(self, val):        
        max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1;
        self.current_frame_number = max * float(val) / self.max_video_position_slider_value
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video.read()
        self.current_frame = frame
        self.draw_image()
        self.update_image()

    def start(self):
        self.tracking_thread = threading.Thread(target = self.tracking.do_tracking, args = 
            (self.video_file_name, self.current_frame_number, self.tracking_flow, self.time_to_stop))
        self.tracking_thread.start()
        self.poll_tracking_flow()

    def quit(self):
        self.time_to_stop.set()    
        self.tracking_thread.join()
        self.root.quit()

    def select_file(self):
        fn = tkFileDialog.askopenfilename()
        if fn: 
            self.video_file_name = fn
            self.on_new_video()
        
    def on_left_mouse_button_down(self, event):  
        self.new_animal_start.x = event.x - self.image_dx
        self.new_animal_start.y = event.y - self.image_dy
        self.adding_new_animal = True

    def on_left_mouse_button_up(self, event):
        if self.adding_new_animal:
            self.adding_new_animal = False
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            self.tracking.add_animal(self.new_animal_start.x / self.image_scale_factor, self.new_animal_start.y / self.image_scale_factor, 
                                     self.new_animal_end.x / self.image_scale_factor, self.new_animal_end.y / self.image_scale_factor)
            self.draw_image()
            self.update_image()            
    
    
    def on_mouse_moved(self, event):
        if self.adding_new_animal:
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            self.draw_image()
            self.update_image()


gui = Gui()
gui.run()
