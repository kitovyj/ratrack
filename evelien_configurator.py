import sys
import numpy as np
import cv2
import math
import Tkinter as Tk
import ttk
import tkFileDialog
import tkMessageBox
from PIL import Image, ImageTk

import geometry
import gui_tools
import evelien_decorator

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
        rows = cols / k
    else:
        rows = height
        cols = rows * k
        
    return cv2.resize(image, (int(cols), int(rows)))

class EvelienConfigurator:

    # possible states definitions
    gs_idle = 1
    gs_setting_regions = 2
    
    state = gs_idle
    
    # initialized in arrange controls    
    image_width = 0
    image_height = 0
    
    image_scale_factor = 0
    image_dx = 0
    image_dy = 0
        
    initial_geometry = '640x480'
    
    controls_created = False
    
    
                
    def __init__(self, config, host, root, frame):
                
        self.config = config        
        self.host = host
        self.decorator = evelien_decorator.EvelienDecorator(config)
        
        self.frame = frame      
        
        # silly tkinter initialization
        self.root = Tk.Toplevel(root)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)                
        self.root.title("Evelien analyzer configuration")
        # self.root.grab_set_global()

        self.root.geometry(self.initial_geometry)
        self.root.bind("<Configure>", self.on_root_resize)
    
        buttons_left_margin = 8
        buttons_top_margin = 10
        buttons_width = 160
        buttons_height = 30
        check_height = 16
        buttons_space = 10    
        label_height = 16
        entry_height = 20
        
        control_y = buttons_top_margin        
        
        self.set_regions_button = gui_tools.create_button(self.root, "Set regions", self.set_regions,
                                                             buttons_left_margin, control_y, buttons_width, buttons_height)                                          
                                                   
        control_y = control_y + buttons_height + buttons_space                                          
        
        gui_tools.create_label(self.root, "Center", 
                               buttons_left_margin, control_y, buttons_width, buttons_height)
                               
        control_y = control_y + label_height + buttons_space                             
        
        coord_entry_space = 8
        coord_entry_width = (buttons_width - coord_entry_space) / 2
        
        control_x = buttons_left_margin
        self.config.circle_center_x_entry = gui_tools.create_entry(self.root, str(self.config.circle_center.x), control_x, control_y, coord_entry_width, entry_height)
        control_x = control_x + coord_entry_width + coord_entry_space
        self.config.circle_center_y_entry = gui_tools.create_entry(self.root, str(self.config.circle_center.y), control_x, control_y, coord_entry_width, entry_height)

        control_y = control_y + entry_height + buttons_space                             
                
        gui_tools.create_label(self.root, "Radius", 
                               buttons_left_margin, control_y, buttons_width, buttons_height)
        
        control_y = control_y + label_height + buttons_space                             
                                       
        self.radius_entry = gui_tools.create_entry(self.root, str(self.config.circle_radius), buttons_left_margin, control_y, buttons_width, entry_height)

        control_y = control_y + entry_height + buttons_space                             

        self.refresh_button = gui_tools.create_button(self.root, "Refresh", self.refresh,
                                                      buttons_left_margin, control_y, buttons_width, buttons_height)                                          

        control_y = control_y + buttons_height + buttons_space                                          

        self.quit_button = gui_tools.create_button(self.root, "Ok", self.quit, 
                                                   buttons_left_margin, control_y, buttons_width, buttons_height)                                            
        
        self.root.update()
        self.arrange_controls(self.root.winfo_width(), self.root.winfo_height())
        
        self.draw_image()
        self.update_image()
        


    def arrange_controls(self, width, height):
        
        left_panel_width = 177
        right_margin = 17
        bottom_margin = 17
        top_panel_height = 0;
        im_container_margin = 10
        
        horz_space = width - left_panel_width - right_margin
        vert_space = height - top_panel_height - bottom_margin
                
        im_container_height = vert_space
        im_container_width = horz_space
         
        self.image_width = im_container_width
        self.image_height = im_container_height
        
        if not (self.frame is None):
            rows, cols = self.frame.shape[:2]            
            (self.image_scale_factor, self.image_dx, self.image_dy) = calculate_scale_factor(cols, rows, self.image_width, self.image_height)
        
        if not self.controls_created:

            self.image_container = Tk.Label(self.root)
            self.image_container.place(x = left_panel_width, y = top_panel_height)
            # have to set fake image to switch 'width' and 'height' interpretation mode
            self.image_container.image = ImageTk.PhotoImage('RGB', (1, 1))
            self.image_container.config(image = self.image_container.image)
            self.image_container.config(relief = Tk.GROOVE, width = im_container_width, height = im_container_height)
            self.image_container.bind('<Button-1>', self.on_left_mouse_button_down)
            self.image_container.bind('<ButtonRelease-1>', self.on_left_mouse_button_up)
            self.image_container.bind('<Motion>', self.on_mouse_moved)


            self.controls_created = True;
            
        else:

            self.image_container.config(width = im_container_width, height = im_container_height)
            
                        
    def set_image(self, container, matrix):        
        img = Image.fromarray(matrix)
        imgtk = ImageTk.PhotoImage(image = img) 
        container.image = imgtk        
        container.configure(image = container.image)

    def update_image(self):
        self.set_image(self.image_container, self.image)            
                
                
    def draw_image(self):

        self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB);                        
        self.image = fit_image(self.image, self.image_width, self.image_height)
        
        green = (0, 255, 0)            

        if not (self.config.circle_center is None):
            center = self.config.circle_center.scaled(self.image_scale_factor)
            if self.state == self.gs_setting_regions:                                                                                 
                cv2.line(self.image, center.as_int_tuple(), 
                         self.evelien_circle_mouse_pos.as_int_tuple(), green)

        self.decorator.decorate_after(None, self.image, self.image_scale_factor)


    # controls events

    def on_root_resize(self, event):    
        if event.widget == self.root:
            self.arrange_controls(event.width, event.height)                    
        
    def refresh(self):                        
        self.config.circle_radius = float(self.radius_entry.var.get())
        self.config.circle_center.x = float(self.config.circle_center_x_entry.get())
        self.config.circle_center.y = float(self.config.circle_center_y_entry.get())
        self.draw_image()
        self.update_image()
        
    def quit(self):     
        self.host.on_configurator_closing()           
        self.root.destroy()

    def set_regions(self):
        self.state = self.gs_setting_regions
        self.config.circle_center = None
        self.set_regions_button.config(relief = Tk.SUNKEN)

    # mouse events    
        
    def on_left_mouse_button_down(self, event):  
        if self.state == self.gs_setting_regions:
            self.evelien_circle_mouse_pos = geometry.Point(event.x - self.image_dx, event.y - self.image_dy)
            self.config.circle_center = self.evelien_circle_mouse_pos.scaled( 1. / self.image_scale_factor ).as_int()
            self.config.circle_radius = 1
            self.radius_entry.var.set(str(self.config.circle_radius))
            self.config.circle_center_x_entry.var.set(str(self.config.circle_center.x))
            self.config.circle_center_y_entry.var.set(str(self.config.circle_center.y))
            

    def on_left_mouse_button_up(self, event):

        if self.state == self.gs_setting_regions:
            self.state = self.gs_idle
            center = self.config.circle_center.scaled(self.image_scale_factor)
            self.config.circle_radius = int(geometry.distance(center.x, center.y, 
                                                       event.x - self.image_dx, event.y - self.image_dy) / self.image_scale_factor)
            self.evelien_circle_mouse_pos.x = event.x - self.image_dx
            self.evelien_circle_mouse_pos.y = event.y - self.image_dy            
            self.radius_entry.var.set(str(self.config.circle_radius))
            self.draw_image()
            self.update_image()
            self.set_regions_button.config(relief = Tk.RAISED)
    
    
    def on_mouse_moved(self, event):
        if self.state == self.gs_setting_regions and not (self.config.circle_center is None):
            center = self.config.circle_center.scaled(self.image_scale_factor)
            self.config.circle_radius = int(geometry.distance(center.x, center.y, 
                                                       event.x - self.image_dx, event.y - self.image_dy) / self.image_scale_factor)
            self.evelien_circle_mouse_pos.x = event.x - self.image_dx
            self.evelien_circle_mouse_pos.y = event.y - self.image_dy            
            self.radius_entry.var.set(str(self.config.circle_radius))
            self.draw_image()
            self.update_image()

