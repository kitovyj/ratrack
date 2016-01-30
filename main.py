import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
import tkFileDialog

from PIL import Image, ImageTk

from tracking import Tracking, Animals, Animal, BodyPart, TrackingFlowElement, curr_cos
import geometry

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

# silly tkinter code to set control size in pixels
def create_button(root, ptext, pcommand, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    button = Tk.Button(f, text = ptext, command = pcommand)    
    button.pack(fill = Tk.BOTH, expand = 1)
    return button

def create_check(root, ptext, initial, pcommand, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    var = Tk.IntVar()
    var.set(initial)
    control = Tk.Checkbutton(f, text = ptext, command = pcommand, variable = var)    
    control.pack(fill = Tk.Y, expand = 1, anchor = Tk.W)
    control.var = var
    return control

class Gui:

    # possible states definitions
    gsNoVideoSelected = 1
    gsNotStarted = 2
    gsRunning = 3
    gsPaused = 4
    
    state = gsNoVideoSelected
    
    tracking_flow = Queue.Queue()    
    
    time_to_stop = threading.Event()
    next_frame_semaphore = threading.Event()
    run_tracking_semaphore = threading.Event()

    current_frame_number = 0

    tracking_thread = 0
    
    #video_file_name = 'videotest.avi'
    #c:\radboud\ratrack\videos\2014-03-22_20-57-44.avi
    video_file_name = 'c:/radboud/ratrack/videos/2014-03-22_20-57-44.avi'
            
    video = 0    
    
    adding_new_animal = False
    new_animal_start = point()
    new_animal_end = point()

    # initialized in arrange controls    
    image_width = 0
    image_height = 0
    
    image_scale_factor = 0
    image_dx = 0
    image_dy = 0
    
    slider = 0
    
    initial_geometry = '1216x800'
    
    controls_created = False
        
    def __init__(self):
        
        # silly tkinter initialization
        self.root = Tk.Tk()
        self.root.withdraw()
        self.root = Tk.Toplevel()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)        
        
        self.root.geometry(self.initial_geometry)
        self.root.bind("<Configure>", self.on_root_resize)
    
        buttons_left_margin = 8
        buttons_top_margin = 25
        buttons_width = 160
        buttons_height = 30
        check_height = 16
        buttons_space = 10        

        control_y = buttons_top_margin        
        self.select_file_button = create_button(self.root, "Select file", self.select_file, 
                                                buttons_left_margin, control_y, buttons_width, buttons_height)                                                
        control_y = control_y + buttons_height + buttons_space                                                
        self.start_button = create_button(self.root, "Run", self.start, 
                                          buttons_left_margin, control_y, buttons_width, buttons_height)                                          
        control_y = control_y + buttons_height + buttons_space                                          
        self.next_button = create_button(self.root, "Next", self.next, 
                                         buttons_left_margin, control_y, buttons_width, buttons_height)                                        
        control_y = control_y + buttons_height + buttons_space                                         
        self.quit_button = create_button(self.root, "Quit", self.quit, 
                                         buttons_left_margin, control_y, buttons_width, buttons_height)                                            
        control_y = control_y + buttons_height + buttons_space                                                                                  
        self.check_show_model = create_check(self.root, "Show model", 1, self.on_show_model,
                                             buttons_left_margin, control_y, buttons_width, buttons_height)
        control_y = control_y + check_height + buttons_space                                                                                  
        self.check_show_posture = create_check(self.root, "Show posture", 1, self.on_show_posture,
                                             buttons_left_margin, control_y, buttons_width, buttons_height)
        
        self.root.update()
        self.arrange_controls(self.root.winfo_width(), self.root.winfo_height())

        self.on_new_video()

    def arrange_controls(self, width, height):
        
        left_panel_width = 177
        right_margin = 17
        bottom_margin = 17
        top_panel_height = 50;
        containers_margin = 10
        
        horz_space = width - left_panel_width - right_margin
        vert_space = height - top_panel_height - bottom_margin
        
        slider_length = horz_space
        
        containers_height = (vert_space - containers_margin) / 2
        containers_width = (horz_space - containers_margin) / 2
         
        self.image_width = containers_width
        self.image_height = containers_height
        
        if self.state != self.gsNoVideoSelected:
            rows, cols = self.current_frame.shape[:2]            
            (self.image_scale_factor, self.image_dx, self.image_dy) = calculate_scale_factor(cols, rows, self.image_width, self.image_height)
        
        if not self.controls_created:

            slider_x = left_panel_width
            slider_y = 0
            self.slider = Tk.Scale(self.root, length = slider_length, from_ = 0, to = 100, 
                 orient = Tk.HORIZONTAL, command = self.on_video_position_changed)                                   
            self.slider.place(x = slider_x, y = slider_y)            

            self.image_container = Tk.Label(self.root)
            self.image_container.place(x = left_panel_width, y = top_panel_height)
            # have to set fake image to switch 'width' and 'height' interpretation mode
            self.image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
            self.image_container.config(image = self.image_container.image)
            self.image_container.config(relief = Tk.GROOVE, width = containers_width, height = containers_height)
            #        self.image_container.config(borderwidth = 1)
            self.image_container.bind('<Button-1>', self.on_left_mouse_button_down)
            self.image_container.bind('<ButtonRelease-1>', self.on_left_mouse_button_up)
            self.image_container.bind('<Motion>', self.on_mouse_moved)

            self.filtered_image_container = Tk.Label(self.root)
            self.filtered_image_container.place(x = left_panel_width + containers_width + containers_margin, y = top_panel_height)
            self.filtered_image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
            self.filtered_image_container.config(image = self.filtered_image_container.image)
            self.filtered_image_container.config(relief = Tk.GROOVE, width = containers_width, height = containers_height)

            self.direction_image_container = Tk.Label(self.root)
            self.direction_image_container.place(x = left_panel_width, y = top_panel_height + containers_height + containers_margin)
            self.direction_image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
            self.direction_image_container.config(image = self.direction_image_container.image)
            self.direction_image_container.config(relief = Tk.GROOVE, width = containers_width, height = containers_height)

            self.controls_created = True;
            
        else:
            self.slider.config(length = slider_length)
            self.image_container.config(width = containers_width, height = containers_height)
            self.filtered_image_container.config(width = containers_width, height = containers_height)
            self.filtered_image_container.place(x = left_panel_width + containers_width + containers_margin)
            self.direction_image_container.config(width = containers_width, height = containers_height)
            self.direction_image_container.place(y = top_panel_height + containers_height + containers_margin)            
                        
    def set_image(self, container, matrix):        
        img = Image.fromarray(matrix)
        imgtk = ImageTk.PhotoImage(image = img) 
        container.image = imgtk        
        container.configure(image = container.image)

    def update_image(self):
        self.set_image(self.image_container, self.current_image)            
    
    def bodypart_center(self, bp):
        c = bp.get_center()
        c.x = c.x * self.image_scale_factor
        c.y = c.y * self.image_scale_factor
        return c;        

    def bodypart_radius(self, bp):
        r = bp.get_radius()
        return r * self.image_scale_factor
    
    def draw_bodypart(self, bp):
        white = (255, 255, 255)
        c = self.bodypart_center(bp)
        cv2.circle(self.current_image, (int(c.x), int(c.y)), 
                   int(self.bodypart_radius(bp)), white)
            
    def draw_animals(self):
        for a in self.tracking.animals.animals:

            if self.check_show_model.var.get():
                            
                self.draw_bodypart(a.back)
                self.draw_bodypart(a.front)
                self.draw_bodypart(a.head)    
                if a.mount != 0:
                    self.draw_bodypart(a.mount)    

            if self.check_show_posture.var.get():
                
                hc = self.bodypart_center(a.head)
                fc = self.bodypart_center(a.front)
                bc = self.bodypart_center(a.back)
                
                hr = self.bodypart_radius(a.head)
                fr = self.bodypart_radius(a.front)
                br = self.bodypart_radius(a.back)

                fhd = geometry.distance(fc.x, fc.y, hc.x, hc.y)
                fbd = geometry.distance(fc.x, fc.y, bc.x, bc.y)
                
                h = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd + hr)
                b = geometry.point_along_a_line(fc.x, fc.y, bc.x, bc.y, fbd + br)
                                
                white = (255, 255, 255)
                green = (0, 255, 0)
                
                cv2.line(self.current_image, (int(b[0]), int(b[1])), 
                         (int(fc.x), int(fc.y)), white)
                cv2.line(self.current_image, (int(fc.x), int(fc.y)), 
                         (int(h[0]), int(h[1])), white)                                    

                cv2.circle(self.current_image, (int(fc.x), int(fc.y)), 2, green)                
                
                ahd = fhd - 4
                if ahd < 0:
                    ahd = 0
                
                arrow_head = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, ahd)
                arrow_line1 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                          arrow_head[0], arrow_head[1], 5)
                arrow_line2 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                          arrow_head[0], arrow_head[1], -5)
                                
                cv2.line(self.current_image, (int(h[0]), int(h[1])), 
                         (int(arrow_line1[0]), int(arrow_line1[1])), white)
                cv2.line(self.current_image, (int(h[0]), int(h[1])), 
                         (int(arrow_line2[0]), int(arrow_line2[1])), white)
                    
                
    def draw_image(self):

        #self.current_image = self.current_frame.copy()        
        self.current_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB);        
          
        #frame = self.current_image

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #roi = frame[171:(171 + 105), 322:(322 + 65)]        

        #hist_size = 16

        #roi_hist = cv2.calcHist([roi], [0, 1, 2], None, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256] )        
        #cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        #frame = cv2.calcBackProject([frame], [0, 1, 2], roi_hist, [0, 256, 0, 256, 0, 256], 2)  
#        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        

#        roi_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256] )        
#        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
#        frame = cv2.calcBackProject([frame], [0, 1], roi_hist, [0, 180, 0, 256], 1)  
        
        
#        self.current_image = frame
        
#        self.current_image = roi
        
        self.current_image = fit_image(self.current_image, self.image_width, self.image_height)
        
        self.draw_animals()
        if self.adding_new_animal:
            cv2.line(self.current_image, (int(self.new_animal_start.x), int(self.new_animal_start.y)), 
                     (int(self.new_animal_end.x), int(self.new_animal_end.y)), (255, 255, 255))                                                 
    
    def poll_tracking_flow(self):
            
        if not self.tracking_flow.empty():
            e = self.tracking_flow.get()
            ret, self.current_frame = self.video.read()

            frame_num = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            
            self.slider.set((self.max_video_position_slider_value * frame_num) / max)            
            
            self.draw_image()
            self.update_image()

            animal = self.tracking.animals.animals[0]
            
            hc = animal.head.get_center()
            fc = animal.front.get_center()
            
            dx = hc.x - fc.x
            dy = hc.y - fc.y
            length = math.sqrt(dx**2 + dy**2)
            cos = - dy / length
            angle = math.acos(cos) * 180 / math.pi            
            if dx < 0:
                angle = -angle
            
            rc = (self.image_width / 2, self.image_height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(rc, angle, 1.0);
            rotated = cv2.warpAffine(self.current_image, rotation_matrix, 
                                     (self.image_width, self.image_height))

            self.set_image(self.direction_image_container, rotated)            
                                   
            width = self.filtered_image_container.winfo_width()
            height = self.filtered_image_container.winfo_height()
            self.set_image(self.filtered_image_container, 
                           fit_image(e.filtered_image, width, height))            

            #rows, cols = e.weights[0].shape[:2]
            #w = np.ones((rows, cols), np.float)
            #w.fill(255)
            #w = np.multiply(w, e.weights[0])
            
            #self.set_image(self.filtered_image_container, 
#                           fit_image(w, self.filtered_image_width, self.filtered_image_height))            
            self.tracking_flow.task_done()                                       

            #print('flush')
            
        self.root.after(30, self.poll_tracking_flow)

    # controls events

    def on_root_resize(self, event):    
        if event.widget == self.root:
            self.arrange_controls(event.width, event.height)                    
        
    def on_new_video(self):
        
        self.video = cv2.VideoCapture(self.video_file_name)

        self.state = self.gsNotStarted
    
        max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self.max_video_position_slider_value = max
        self.slider["to"] = self.max_video_position_slider_value
                
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

    def on_video_position_changed(self, val):
        if self.state != self.gsNotStarted:
            return
        max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1;
        self.current_frame_number = max * float(val) / self.max_video_position_slider_value
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video.read()
        self.current_frame = frame
        self.draw_image()
        self.update_image()

    def start(self):
        if self.state == self.gsRunning:
            self.start_button["text"] = "Run"
            self.run_tracking_semaphore.clear();
            self.state = self.gsPaused
        elif self.state == self.gsPaused:
            self.start_button["text"] = "Pause"
            self.run_tracking_semaphore.set();
            self.next_frame_semaphore.set();
            self.state = self.gsRunning
        else:            
            self.start_button["text"] = "Pause"
            self.run_tracking_semaphore.set();
            self.tracking_thread = threading.Thread(target = self.tracking.do_tracking, args = 
                (self.video_file_name, self.current_frame_number, self.tracking_flow, self.time_to_stop, self.next_frame_semaphore, self.run_tracking_semaphore))
           #        self.tracking.do_tracking(self.video_file_name, self.current_frame_number, self.tracking_flow, self.time_to_stop);
            self.tracking_thread.start()
            self.poll_tracking_flow()
            self.state = self.gsRunning

    def next(self):
        self.next_frame_semaphore.set()

    def quit(self):
        if self.tracking_thread != 0:
            self.time_to_stop.set()        
            self.tracking_thread.join()
        self.root.quit()

    def select_file(self):
        fn = tkFileDialog.askopenfilename()
        if fn: 
            self.video_file_name = fn
            self.on_new_video()

    def on_show_model(self):
        if self.state != self.gsNoVideoSelected:
            self.draw_image()
            self.update_image()            

    def on_show_posture(self):
        if self.state != self.gsNoVideoSelected:
            self.draw_image()
            self.update_image()            

    # mouse events    
        
    def on_left_mouse_button_down(self, event):  
        self.new_animal_start.x = event.x - self.image_dx
        self.new_animal_start.y = event.y - self.image_dy
        self.adding_new_animal = True

    def on_left_mouse_button_up(self, event):
        if self.adding_new_animal:
            self.adding_new_animal = False
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            a = self.tracking.add_animal(self.new_animal_start.x / self.image_scale_factor, self.new_animal_start.y / self.image_scale_factor, 
                                     self.new_animal_end.x / self.image_scale_factor, self.new_animal_end.y / self.image_scale_factor)
            a.best_fit(self.current_frame)
            self.draw_image()
            self.update_image()            
    
    
    def on_mouse_moved(self, event):
        if self.adding_new_animal:
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            self.draw_image()
            self.update_image()


# main()

gui = Gui()
gui.run()
