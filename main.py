import sys
import numpy as np
import cv2
import math
import threading
import Queue
import os

import Tkinter as Tk
import ttk
import tkFileDialog
import tkMessageBox

from PIL import Image, ImageTk

from tracking import *
from geometry import *

from analyzers.intruder import *
from analyzers.evelien import *

from gui_tools import *

import config_serialization

# tkinter layout management : http://zetcode.com/gui/tkinter/layout/                        
        
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
        rows = max(1, cols / k)
    else:
        rows = height
        cols = max(1, rows * k)
        
    return cv2.resize(image, (int(cols), int(rows)))

class TextBoxLogger:
    def __init__(self, text_box):
        self.text_box = text_box
    def log(self, message):
        self.text_box.insert(Tk.END, message + '\n')
        self.text_box.yview(Tk.END)    

class QueueLogger:
    def __init__(self, queue):
        self.queue = queue
    def log(self, message):
        self.queue.put(message)

class Gui:

    # possible states definitions
    gs_no_video_selected = 1
    gs_not_started = 2
    gs_running = 3
    gs_paused = 4
    gs_adding_animal = 5
    
    state = gs_no_video_selected

    tracker_messages_queue = Queue.Queue(200)    
    
    tracking_flow = Queue.Queue(20)    
    
    time_to_stop = threading.Event()
    next_frame_semaphore = threading.Event()
    run_tracking_semaphore = threading.Event()

    current_frame_number = 0
    
    current_animal_positions = []

    tracking_thread = 0
    
    #video_file_name = 'videotest.avi'
    #c:\radboud\ratrack\videos\2014-03-22_20-57-44.avi
    video_file_name = 'c:/radboud/ratrack/videos/2014-03-22_20-57-44.avi'
            
    video = 0    
    
    new_animal_start = geometry.Point()
    new_animal_end = geometry.Point()

    # initialized in arrange controls    
    image_width = 0
    image_height = 0
    
    image_scale_factor = 0
    image_dx = 0
    image_dy = 0
    
    slider = 0
    
    initial_geometry = '1216x800'
    
    controls_created = False
    
    writer = 0
    
    debug_frame_containers = dict()
    
    draw_one_frame = False
    
    # evelien    
    evelien_circle_center = None    
    
    class AnalyzerState:
        def __init__(self, factory):
            self.factory = factory
            self.configuration = factory.create_configuration()
            self.decorator = factory.create_decorator(self.configuration)
    
    analyzer_states = [ None, AnalyzerState(intruder.factory()), AnalyzerState(evelien.factory()) ]        
    # active analyzer
    analyzer = None
    
    got_first_tracking_element = False
            
    def __init__(self):
        
        # silly tkinter initialization
        self.root = Tk.Tk()
        self.root.withdraw()
        self.root = Tk.Toplevel()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)        

        self.root.title("Rat tracking tool")        
        self.root.geometry(self.initial_geometry)
        self.root.bind("<Configure>", self.on_root_resize)    
    
        buttons_left_margin = 8
        buttons_top_margin = 15
        buttons_width = 160
        buttons_height = 35
        check_height = 23
        radio_height = 23
        label_height = 20
        buttons_space = 5       
        
        # create main menu
        
        self.menu = Tk.Menu(self.root)
        
        misc = Tk.Menu(self.menu, tearoff = 0)
        misc.add_command(label = 'Source frame screenshot', command = self.on_source_screenshot)
        misc.add_command(label = 'Drawn frame screenshot', command = self.on_drawn_frame_screenshot)
        misc.add_command(label = 'Debug frame screenshot', command = self.on_debug_frame_screenshot)
        
        self.menu.add_cascade(label = "Misc", menu = misc)                

        self.root.config(menu = self.menu)
        
        control_y = buttons_top_margin        
        self.select_file_button = gui_tools.create_button(self.root, "Select file", self.select_file, 
                                                          buttons_left_margin, control_y, buttons_width, buttons_height)                                                
        control_y = control_y + buttons_height + buttons_space                                                
        self.calc_bg_button = gui_tools.create_button(self.root, "Calculate background", self.calculate_background, 
                                                      buttons_left_margin, control_y, buttons_width, buttons_height)                                                
        control_y = control_y + buttons_height + buttons_space                                                
        self.start_button = gui_tools.create_button(self.root, "Run", self.start, 
                                                    buttons_left_margin, control_y, buttons_width, buttons_height)                                          
        control_y = control_y + buttons_height + buttons_space                                          
        self.next_button = gui_tools.create_button(self.root, "Next", self.next, 
                                                   buttons_left_margin, control_y, buttons_width, buttons_height)                                        
        control_y = control_y + buttons_height + buttons_space                                         
        self.next_button = gui_tools.create_button(self.root, "Configure tracker", self.configure_tracker, 
                                                   buttons_left_margin, control_y, buttons_width, buttons_height)                                        
        control_y = control_y + buttons_height + buttons_space                                         
        self.quit_button = gui_tools.create_button(self.root, "Quit", self.quit, 
                                                   buttons_left_margin, control_y, buttons_width, buttons_height)                                            
        control_y = control_y + buttons_height + buttons_space                                                                                  
        self.check_show_model = gui_tools.create_check(self.root, "Show model", 1, self.on_show_model,
                                                       buttons_left_margin, control_y, buttons_width, buttons_height)
        control_y = control_y + check_height + buttons_space                                                                                  
        self.check_show_posture = gui_tools.create_check(self.root, "Show posture", 1, self.on_show_posture,
                                                         buttons_left_margin, control_y, buttons_width, buttons_height)
        control_y = control_y + check_height + buttons_space                                                                                  
        self.check_show_debug = gui_tools.create_check(self.root, "Show debug", 1, self.on_show_debug,
                                                         buttons_left_margin, control_y, buttons_width, buttons_height)
        control_y = control_y + check_height + buttons_space                                                                                  
        self.check_show_rotated = gui_tools.create_check(self.root, "Show rotated frames", 1, self.on_show_rotated,
                                                         buttons_left_margin, control_y, buttons_width, buttons_height)

        control_y = control_y + check_height + buttons_space                                                                                  
        self.check_show_analyzer_data = gui_tools.create_check(self.root, "Show analyzer data", 1, self.on_show_analyzer_data,
                                                               buttons_left_margin, control_y, buttons_width, buttons_height)

        control_y = control_y + check_height + buttons_space                                                                                  

        gui_tools.create_label(self.root, "Animal model", 
                               buttons_left_margin, control_y, buttons_width, buttons_height)

        control_y = control_y + label_height + buttons_space                             

        self.animal_model = Tk.IntVar()
        self.animal_model.set(tracking.Animal.Configuration.model_with_drive)

        for m in ((tracking.Animal.Configuration.model_normal, 'Normal'), 
                  (tracking.Animal.Configuration.model_with_drive, 'With a drive')):
            gui_tools.create_radio(self.root, m[1], self.animal_model, m[0],
                                   buttons_left_margin, control_y, buttons_width, buttons_height)
            control_y = control_y + radio_height + buttons_space
                                    
        control_y = control_y + buttons_space
       
        gui_tools.create_label(self.root, "Analyzers", 
                               buttons_left_margin, control_y, buttons_width, buttons_height)
       
        control_y = control_y + label_height + buttons_space                             

        self.analyzer_index = Tk.IntVar()
        self.analyzer_index.set(0)

        for i, a in enumerate(self.analyzer_states):
            if a is None:                
                gui_tools.create_radio(self.root, 'None', self.analyzer_index, i,
                                       buttons_left_margin, control_y, buttons_width, buttons_height)
            else:
                gui_tools.create_radio(self.root, a.factory.name, self.analyzer_index, i,
                                       buttons_left_margin, control_y, buttons_width, buttons_height)
            control_y = control_y + radio_height + buttons_space
                                    
        self.configure_analyzer_button = gui_tools.create_button(self.root, "Configure analyzer", self.configure_analyzer,
                                                                 buttons_left_margin, control_y, buttons_width, buttons_height)                                          
                                                   
        control_y = control_y + buttons_height + buttons_space                                          
        
        self.root.update()
        self.arrange_controls(self.root.winfo_width(), self.root.winfo_height())

        self.tracking_config = config_serialization.load_tracking_config('tracking.cfg')
        self.animal_config = config_serialization.load_animal_config('animal.cfg')

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
        
        if self.state != self.gs_no_video_selected:
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

            self.debug_tabs = gui_tools.create_tabs(self.root, left_panel_width + containers_width + containers_margin, top_panel_height, containers_width, containers_height)
            
            self.direction_image_container = Tk.Label(self.root)
            self.direction_image_container.place(x = left_panel_width, y = top_panel_height + containers_height + containers_margin)
            self.direction_image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
            self.direction_image_container.config(image = self.direction_image_container.image)
            self.direction_image_container.config(relief = Tk.GROOVE, width = containers_width, height = containers_height)


            self.messages_tabs = gui_tools.create_tabs(self.root, left_panel_width + containers_width + containers_margin, 
                                                       top_panel_height + containers_height + containers_margin, containers_width, containers_height)
                                                       
            page = ttk.Frame(self.messages_tabs[1])            
            self.tracker_messages = Tk.Text(page)    
            self.tracker_messages.pack(expand = 1, fill = "both")            
            self.messages_tabs[1].add(page, text = "Tracker")

            page = ttk.Frame(self.messages_tabs[1])
            self.analyzer_messages = Tk.Text(page)    
            self.analyzer_messages.pack(expand = 1, fill = "both")            
            self.messages_tabs[1].add(page, text = "Analyzer")
                                                       
            self.controls_created = True;
            
        else:
            self.slider.config(length = slider_length)
            self.image_container.config(width = containers_width, height = containers_height)

            self.debug_tabs[0].config(width = containers_width, height = containers_height)
            self.debug_tabs[0].place(x = left_panel_width + containers_width + containers_margin)
                        
            self.direction_image_container.config(width = containers_width, height = containers_height)
            self.direction_image_container.place(y = top_panel_height + containers_height + containers_margin)            
            
            self.messages_tabs[0].place(x = left_panel_width + containers_width + containers_margin, y = top_panel_height + containers_height + containers_margin)
            self.messages_tabs[0].config(width = containers_width, height = containers_height)
            
                        
    def set_image(self, container, matrix):        
        img = Image.fromarray(matrix)
        imgtk = ImageTk.PhotoImage(image = img) 
        container.image = imgtk        
        container.configure(image = container.image)

    def update_image(self):
        self.set_image(self.image_container, self.current_image)            
    
    def project(self, pos):
        r = geometry.Point(pos.x * self.image_scale_factor, pos.y * self.image_scale_factor)
        return r;        

    def scaled_radius(self, bp):
        r = bp.get_radius()
        return r * self.image_scale_factor
    
    def draw_bodypart(self, bp, pos, color = (255, 255, 255)):
        pos = self.project(pos)
        cv2.circle(self.current_image, (int(pos.x), int(pos.y)), 
                   int(self.scaled_radius(bp)), color)

    def draw_head(self, head, pos, front_pos, color = (255, 255, 255)):
        
        if not head.triangle:
            self.draw_bodypart(head, pos, color)
            return
        
        hc = self.project(pos)
        fc = self.project(front_pos)
        hr = self.scaled_radius(head)
        fhd = geometry.distance(fc.x, fc.y, hc.x, hc.y)        

        side = math.sqrt(3.) * hr
        height = 3. * hr / 2.

        top = geometry.Point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd + hr)
        bottom = geometry.Point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd - height / 2)
        
  
        left = geometry.Point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                    bottom[0], bottom[1], side / 2)
        right = geometry.Point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                    bottom[0], bottom[1], -side / 2)
        
        cv2.line(self.current_image, (int(top[0]), int(top[1])), 
                 (int(left[0]), int(left[1])), color)
        cv2.line(self.current_image, (int(top[0]), int(top[1])), 
                 (int(right[0]), int(right[1])), color)
        cv2.line(self.current_image, (int(left[0]), int(left[1])), 
                 (int(right[0]), int(right[1])), color)            
            
    def draw_animals(self):
        
        for ap in self.current_animal_positions:
            
            white = (255, 255, 255)
            green = (0, 255, 0)            
            red = (255, 0, 0)            
            yellow = (255, 255, 0)
            
            a = ap[0]
            p = ap[1]
                        
            if self.check_show_model.var.get():

                # find min and max values

                max_val = -1
                min_val = -1
                
                for v in p.backbone:
                    if v.value > max_val:
                        max_val = v.value
                    if min_val == -1 or v.value < min_val:
                        min_val = v.value
                val_delta = max_val - min_val
                        
                for idx, v in enumerate(p.backbone):
                    vc = self.project(v.center)
                    if idx > 0:
                        pvc = self.project(p.backbone[idx - 1].center)                
                        cv2.line(self.current_image, (int(pvc.x), int(pvc.y)), 
                             (int(vc.x), int(vc.y)), white)                       
                        
                for idx, v in enumerate(p.backbone):
                    vc = self.project(v.center)
                    if val_delta != 0:                        
                        intensity = 255 - 255 * (v.value - min_val) / val_delta
                    else:
                        intensity = 255
                    color = (255, intensity, intensity)
                    r = 2
                    if idx == p.central_vertebra_index:
                        r = 4
                    cv2.circle(self.current_image, (int(vc.x), int(vc.y)), r, color, -1)                
                    if idx == len(p.backbone) - 1:
                        cv2.circle(self.current_image, (int(vc.x), int(vc.y)), 6, green)                
                         
                
    def draw_image(self):

        self.current_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB);        
        
        self.current_image = fit_image(self.current_image, self.image_width, self.image_height)

        analyzer_state = self.analyzer_states[self.analyzer_index.get()]
        
        if not (analyzer_state is None) and self.check_show_analyzer_data.var.get():    
            analyzer_state.decorator.decorate_before(self.analyzer, self.current_image, self.image_scale_factor)
        
        self.draw_animals()
        if self.state == self.gs_adding_animal:
            cv2.line(self.current_image, (int(self.new_animal_start.x), int(self.new_animal_start.y)), 
                     (int(self.new_animal_end.x), int(self.new_animal_end.y)), (255, 255, 255))                                                 

        if (not (analyzer_state is None)) and self.check_show_analyzer_data.var.get():    
            analyzer_state.decorator.decorate_after(self.analyzer, self.current_image, self.image_scale_factor)
        
    
    def poll_tracking_flow(self):

        if self.time_to_stop.isSet():
            return

        while not self.tracker_messages_queue.empty():
            e = self.tracker_messages_queue.get()            
            self.tracker_messages.insert(Tk.END, e + '\n')

        self.tracker_messages.yview(Tk.END)    

        e = 0        
        max_elements_to_get = 10
        while not self.tracking_flow.empty() and max_elements_to_get > 0:
            e = self.tracking_flow.get()
            if self.got_first_tracking_element:
                ret, self.current_frame = self.video.read()            
            else:
                self.got_first_tracking_element = True
            if not (self.analyzer is None):                
                self.analyzer.analyze(e)
            if self.tracking.finished:
                self.analyzer.on_finished()
            max_elements_to_get = max_elements_to_get - 1

        if e != 0:
                    
            frame_num = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            
            self.slider.set((self.max_video_position_slider_value * frame_num) / max)            
            self.current_animal_positions = e.positions
            self.draw_image()

            '''
            if self.writer == 0:
                rows, cols = self.current_image.shape[:2]                                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter('output.avi', -1 , 20.0, (cols, rows))        
            
            
            self.writer.write(self.current_image)            
            '''
            
            self.update_image()

            '''
            if self.check_show_rotated.var.get():

                p = e.positions[0][1]
            
                hc = p.head
                fc = p.front
            
                dx = hc.x - fc.y
                dy = hc.x - fc.y
                length = math.sqrt(dx**2 + dy**2)
                if length != 0:
                    cos = - dy / length
                    angle = math.acos(cos) * 180 / math.pi            
                    if dx < 0:
                        angle = -angle
                else:
                    angle = 0
            
                rc = (self.image_width / 2, self.image_height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(rc, angle, 1.0);
                rotated = cv2.warpAffine(self.current_image, rotation_matrix, 
                                         (self.image_width, self.image_height))

                self.set_image(self.direction_image_container, rotated)            
            '''
                             
            '''
            width = self.filtered_image_container.winfo_width()
            height = self.filtered_image_container.winfo_height()
            self.set_image(self.filtered_image_container, 
                           fit_image(e.filtered_image, width, height))            
            '''
            
            #w = np.ones((rows, cols), np.float)
            #w.fill(255)
            #w = np.multiply(w, e.weights[0])
            if self.check_show_debug.var.get():

                for df in e.debug_frames:
                    (name, frame) = df                
                    if name in self.debug_frame_containers:
                        frame_container = self.debug_frame_containers[name]
                    else:
                        page = ttk.Frame(self.debug_tabs[1])
                        image_container = Tk.Label(page)
                        image_container.pack(expand = 1, fill = "both")
                        self.debug_tabs[1].add(page, text = name)
                        frame_container = (page, image_container, None)
                        self.debug_frame_containers[name] = frame_container

                    (page, image_container, old_frame) = frame_container

                    width = self.debug_tabs[1].winfo_width()
                    height = self.debug_tabs[1].winfo_height()
                    self.set_image(image_container, fit_image(frame, width, height))            
                    
                    self.debug_frame_containers[name] = (page, image_container, frame)
                    

            self.draw_one_frame = False
                                
            
            #self.set_image(self.filtered_image_container, 
#                           fit_image(w, cols, rows))            
            
            
            #self.tracking_flow.task_done()                                       

            #print('flush')
            
        if self.state != self.gs_paused or self.draw_one_frame:
            self.root.after(10, self.poll_tracking_flow)

    # controls events

    def on_root_resize(self, event):    
        if event.widget == self.root:
            self.arrange_controls(event.width, event.height)                    
        
    def on_new_video(self):
        
        self.video = cv2.VideoCapture(self.video_file_name)

        self.state = self.gs_not_started
    
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
                
        self.tracking = tracking.Tracking(self.video_file_name, self.tracking_config, QueueLogger(self.tracker_messages_queue))

        (self.image_scale_factor, self.image_dx, self.image_dy) = calculate_scale_factor(cols, rows, self.image_width, self.image_height)
                
        self.draw_image()
        self.update_image()
        
    def run(self):
        self.root.mainloop()        
        self.root.destroy()
        self.video.release() 

    def get_bg_file_name(self):
        bg_file_name = os.path.splitext(self.video_file_name)[0]
        return bg_file_name + '-bg.tiff'

    def on_video_position_changed(self, val):
        if self.state != self.gs_not_started:
            return
        max = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1;
        self.current_frame_number = max * float(val) / self.max_video_position_slider_value
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video.read()
        self.current_frame = frame
        self.draw_image()
        self.update_image()

    def start(self):
        if self.state == self.gs_running:
            self.start_button["text"] = "Run"
            self.run_tracking_semaphore.clear();
            self.state = self.gs_paused
        elif self.state == self.gs_paused:
            self.start_button["text"] = "Pause"
            self.run_tracking_semaphore.set();
            self.next_frame_semaphore.set();
            self.state = self.gs_running
            self.root.after(1, self.poll_tracking_flow)            
        else:            
            self.start_button["text"] = "Pause"
            self.run_tracking_semaphore.set();            
            bg = cv2.imread(self.get_bg_file_name())            
            self.tracking_thread = threading.Thread(target = self.tracking.do_tracking, args = 
                (bg, self.current_frame_number, self.tracking_flow, self.time_to_stop, self.next_frame_semaphore, self.run_tracking_semaphore))
           #        self.tracking.do_tracking(self.video_file_name, self.current_frame_number, self.tracking_flow, self.time_to_stop);
            analyzer_state = self.analyzer_states[self.analyzer_index.get()]

            if not (analyzer_state is None):
                self.analyzer = analyzer_state.factory.create_analyzer(analyzer_state.configuration, TextBoxLogger(self.analyzer_messages))
            else:
                self.analyzer = None
                
            self.tracking_thread.start()
            self.poll_tracking_flow()
            self.state = self.gs_running

    def next(self):
        self.draw_one_frame = True
        self.next_frame_semaphore.set()
        self.root.after(1, self.poll_tracking_flow)            

    def quit(self):
        if self.tracking_thread != 0:

            self.time_to_stop.set()

            # clear the queue
            while not self.tracker_messages_queue.empty():
                self.tracker_messages_queue.get()
                self.tracker_messages_queue.task_done()                                       
            
            while not self.tracking_flow.empty():
                self.tracking_flow.get()
                self.tracking_flow.task_done()                                       

            self.next_frame_semaphore.set()
            
#            self.tracking_flow.task_done()
            self.tracking_thread.join()
            
        if self.writer != 0:
            self.writer.release()
        self.root.quit()

    def select_file(self):
        fn = tkFileDialog.askopenfilename()
        if fn: 
            self.video_file_name = fn
            self.on_new_video()
            
    def calculate_background(self):
        if tkMessageBox.askyesno('Calculate background', 'Calculate background for the loaded video(it can take a long time)?'):        
            bg = self.tracking.calculate_background()                
            cv2.imwrite(self.get_bg_file_name(), bg)

    def configure_tracker(self):
        state = self.analyzer_states[self.analyzer_index.get()]        
        state.factory.create_configurator(state.configuration, self, self.root, self.current_frame)        

    def configure_analyzer(self):
        state = self.analyzer_states[self.analyzer_index.get()]        
        state.factory.create_configurator(state.configuration, self, self.root, self.current_frame)        
            
    def on_configurator_closing(self):
        self.draw_image()
        self.update_image()                   

    def on_show_model(self):
        if self.state != self.gs_no_video_selected:
            self.draw_image()
            self.update_image()            

    def on_show_posture(self):
        if self.state != self.gs_no_video_selected:
            self.draw_image()
            self.update_image()            

    def on_show_debug(self):
        if self.state != self.gs_no_video_selected:
            self.draw_image()
            self.update_image()            

    def on_show_rotated(self):
        if self.state != self.gs_no_video_selected:
            self.draw_image()
            self.update_image()            

    def on_show_analyzer_data(self):
        if self.state != self.gs_no_video_selected:
            self.draw_image()
            self.update_image()            
    
    # menu events
    
    def on_source_screenshot(self):
        cv2.imwrite('source_frame.png', self.current_frame)

    def on_drawn_frame_screenshot(self):
        image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR);   
        cv2.imwrite('drawn_frame.png', image)

    def on_debug_frame_screenshot(self):
        curently_selected = self.debug_tabs[1].tab(self.debug_tabs[1].select(), "text")
        (page, image_container, frame) = self.debug_frame_containers[curently_selected]
        cv2.imwrite('debug_frame.png', frame)

    # mouse events    
        
    def on_left_mouse_button_down(self, event):  
        self.new_animal_start.x = event.x - self.image_dx
        self.new_animal_start.y = event.y - self.image_dy
        self.state = self.gs_adding_animal
            

    def on_left_mouse_button_up(self, event):
        if self.state == self.gs_adding_animal:
            self.state = self.gs_not_started
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            
            model = self.animal_model.get()
            if model == tracking.Animal.Configuration.model_with_drive:
                self.animal_model.set(tracking.Animal.Configuration.model_normal)
            self.animal_config.model = model
            
            a = self.tracking.add_animal(self.new_animal_start.x / self.image_scale_factor, self.new_animal_start.y / self.image_scale_factor, 
                                         self.new_animal_end.x / self.image_scale_factor, self.new_animal_end.y / self.image_scale_factor, 
                                         self.animal_config)
#            a.best_fit(self.current_frame)
            self.current_animal_positions = self.tracking.get_animal_positions()
            self.draw_image()
            self.update_image()            
    
    
    def on_mouse_moved(self, event):
        if self.state == self.gs_adding_animal:
            self.new_animal_end.x = event.x - self.image_dx
            self.new_animal_end.y = event.y - self.image_dy
            self.draw_image()
            self.update_image()


# main()

gui = Gui()
gui.run()
