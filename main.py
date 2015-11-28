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
        return float(dst_width) / src_width
    else:
        return float(dst_height) / src_height        

class Gui:
    
    tracking_flow = Queue.Queue()    
    
    time_to_stop = threading.Event()

    current_frame_number = 0

    tracking_thread = 0
    
    video_file_name = 'videotest.avi'
    
    adding_new_animal = False
    new_animal_start = point()
    new_animal_end = point()
    
    max_width = 500
    max_height = 400
    
    scale_factor = 1
        
    def __init__(self):
    
        self.root = Tk.Tk()
        self.root.geometry('800x600')
    
        self.image_container = Tk.Label(self.root)
        self.image_container.place(x = 100, y = 20)
        self.image_container.bind('<Button-1>', self.on_left_mouse_button_down)
        self.image_container.bind('<ButtonRelease-1>', self.on_left_mouse_button_up)
        self.image_container.bind('<Motion>', self.on_mouse_moved)

        self.filtered_image_container = Tk.Label(self.root)
        self.filtered_image_container.place(x = 430, y = 20)
        
        self.quit_button = Tk.Button(self.root, text = "Quit", command = self.quit)
        self.quit_button.place(x = 5, y = 100)
        self.start_button = Tk.Button(self.root, text = "Start", command = self.start)
        self.start_button.place(x = 5, y = 50)
        self.select_file_button = Tk.Button(self.root, text = "Select file", command = self.select_file)
        self.select_file_button.place(x = 5, y = 10)

        self.on_new_video()
        
        self.max_video_position_slider_value = 100
        self.slider = Tk.Scale(self.root, length = 300, from_ = 0, to = self.max_video_position_slider_value, 
                  orient = Tk.HORIZONTAL, command = self.on_video_position_changed)                  
        self.slider.pack(side = Tk.BOTTOM)                
        
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
        
        self.scale_factor = calculate_scale_factor(cols, rows, self.max_width, self.max_height)

        self.draw_image()
        self.update_image()
        
    def run(self):
        
        self.root.mainloop()
        
        self.root.destroy()
        self.video.release() 
            
    def set_image(self, container, matrix):
        img = Image.fromarray(matrix)
        imgtk = ImageTk.PhotoImage(image = img) 
        container.configure(image = imgtk)
        container.image = imgtk        

    def update_image(self):
        self.set_image(self.image_container, self.current_image)            
    
    def draw_bodypart(self, bp):
        c = bp.get_center()
        r = bp.get_radius()
        cv2.circle(self.current_image, (int(c.x * self.scale_factor), int(c.y * self.scale_factor)), 
                   int(r * self.scale_factor), (255, 255, 255))
            
    def draw_animals(self):
        for a in self.tracking.animals.animals:
            self.draw_bodypart(a.back)
            self.draw_bodypart(a.front)
            self.draw_bodypart(a.head)    
                
    def draw_image(self):
        self.current_image = self.current_frame.copy()

        rows, cols = self.current_image.shape[:2]
        
        k = float(cols) / rows
                
        if k > float(self.max_width) / self.max_height:
            cols = self.max_width
            rows = cols / k
        else:
            rows = self.max_height
            cols = rows * k

        self.current_image = cv2.resize(self.current_image, (int(cols), int(rows)))
        
        self.draw_animals()
        if self.adding_new_animal:
            cv2.line(self.current_image, (self.new_animal_start.x, self.new_animal_start.y), (self.new_animal_end.x, self.new_animal_end.y), (255, 255, 255))                
    
    def poll_tracking_flow(self):
            
        if not self.tracking_flow.empty():
            e = self.tracking_flow.get()
            ret, self.current_frame = self.video.read()
            self.draw_image()
            self.update_image()         
            
            self.set_image(self.filtered_image_container, e.filtered_image)            
            
        self.root.after(100, self.poll_tracking_flow)

    def on_video_position_changed(self, val):        
        max = self.video.get(cv2.CAP_PROP_FRAME_COUNT);
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
        self.new_animal_start.x = event.x
        self.new_animal_start.y = event.y
        self.adding_new_animal = True

    def on_left_mouse_button_up(self, event):
        if self.adding_new_animal:
            self.new_animal_end.x = event.x
            self.new_animal_end.y = event.y
            self.adding_new_animal = False
            self.tracking.add_animal(self.new_animal_start.x / self.scale_factor, self.new_animal_start.y / self.scale_factor, 
                                     self.new_animal_end.x / self.scale_factor, self.new_animal_end.y / self.scale_factor)
            self.draw_image()
            self.update_image()            
    
    
    def on_mouse_moved(self, event):
        if self.adding_new_animal:
            self.new_animal_end.x = event.x
            self.new_animal_end.y = event.y
            self.draw_image()
            self.update_image()


gui = Gui()
gui.run()
