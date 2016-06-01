import sys
import numpy as np
import cv2
import math
import threading
import Queue
import pdb
import time
import csv
from skimage import morphology

from geometry import *
from vertebra import *

import tracking_boundary_alignment
import tracking_central_alignment
import bwmorph_thin

class AnimalPosition:
    def __init__(self):
        self.backbone = []
        
class Animal:
        
    class Configuration:
        model_normal = 0
        model_with_drive = 1        
        max_body_length = 60
        max_body_width = 30
        min_body_width = 30        
        front_min_value_coeff = 100
        back_min_value_coeff = 100            
        def __init__(self):
            self.model = self.model_normal
    
    class VertebraPosition:
        def __init__(self, x, y, value):
            self.center = geometry.Point(x, y)            
            self.value = value
                                                
    backbone = []            
        
    contours = None
        
    def __init__(self, host, id, start_x, start_y, end_x, end_y, config = Configuration()):
        
        self.host = host
        self.id = id
        self.config = config
        
        self.boundary_aligner = tracking_boundary_alignment.BoundaryAligner(self)
        self.central_aligner = tracking_central_alignment.CentralAligner(self)        
        
        self.scaled_max_body_length = config.max_body_length * self.host.scale_factor
        self.scaled_max_width = self.config.max_body_width * self.host.scale_factor
        self.scaled_min_width = self.config.min_body_width * self.host.scale_factor
        
        self.max_vertebra = config.max_body_length / self.host.config.vertebra_length
        
        if config.model == self.config.model_with_drive:
            self.max_vertebra = (config.max_body_length * 1.3) / self.host.config.vertebra_length
        else:
            self.max_vertebra = config.max_body_length / self.host.config.vertebra_length
        
        self.max_vertebra = int(round(self.max_vertebra))
                        
        border = self.host.config.skeletonization_border        
        start_x =  start_x * host.scale_factor + border
        start_y =  start_y * host.scale_factor + border
        end_x =  end_x * host.scale_factor + border
        end_y =  end_y * host.scale_factor + border
                
        max_dist = geometry.distance(end_x, end_y, start_x, start_y)
        
        dist = 0
        
        if max_dist < self.host.config.vertebra_length * 2:
            max_dist = self.host.config.vertebra_length * 2

        self.backbone = []
        
        while dist <= max_dist:            
            mount_position = geometry.point_along_a_line(end_x, end_y, start_x, start_y, dist)            
            self.backbone.append(Vertebra(mount_position[0], mount_position[1], 0))            
            dist = dist + self.host.config.vertebra_length            
            
        self.central_vertebra_index = 1
            
        mass_center_filter_size = int(round(self.scaled_max_width * 2))
        mass_center_filter_center = mass_center_filter_size / 2
        self.mass_center_filter = np.ones((mass_center_filter_size, mass_center_filter_size), np.float)
                
        for i in xrange(0, mass_center_filter_size):
            for j in xrange(0, mass_center_filter_size):
                if i != mass_center_filter_center or j != i:
                    dist = geometry.distance(i, j, mass_center_filter_center, mass_center_filter_center)
                    if dist < self.scaled_max_width:
                        self.mass_center_filter[i, j] = 1. / (dist)
                    else:
                        self.mass_center_filter[i, j] = 0
        
    def get_position(self):        
        r = AnimalPosition()
        r.backbone = []
        border = self.host.config.skeletonization_border
        for v in self.backbone:
            r.backbone.append(self.VertebraPosition((v.center.x - border) / self.host.scale_factor, 
                                                    (v.center.y - border) / self.host.scale_factor, v.value))
        r.central_vertebra_index = self.central_vertebra_index            
        return r
                                    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)        

    def set_weights_no_mount(self, matrix, weight):        
        if not (self.contours is None):
            rows, cols = self.contours.shape[:2]            
            np.multiply(matrix, self.contours[1:rows - 1, 1:cols - 1], matrix)        

    def set_weights(self, matrix, weight, mount_weight):
        self.set_weights_no_mount(matrix, weight)                        

    def set_sure_area_1(self, backbone, matrix, value, start_radius, end_radius):

        max_val = -1
        min_val = -1
                
        for v in backbone:
            if v.value > max_val:
                max_val = v.value
            if min_val == -1 or v.value < min_val:
                min_val = v.value
     
        val_delta = max_val - min_val

        radii_delta = start_radius - end_radius

        left = []
        right = []
        for idx, v in enumerate(backbone):

            if val_delta != 0:
                r = end_radius + radii_delta * (v.value - min_val) / val_delta
            else:
                r = start_radius
                
            if idx == 0:
                s = v.center
                e = backbone[idx + 1].center
            elif idx == len(backbone) - 1:
                s = backbone[idx - 1].center
                e = v.center
            else:
                prev = backbone[idx - 1].center
                next = backbone[idx + 1].center
                dx = next.x - prev.x
                dy = next.y - prev.y
                s = v.center
                e = geometry.Point(v.center.x + dx, v.center.y + dy)
        
            p = geometry.point_along_a_perpendicular_p(s, e, v.center, r)
            left.append(list(p))
            p = geometry.point_along_a_perpendicular_p(s, e, v.center, -r)
            right.append(list(p))
            
        cv2.fillConvexPoly(matrix, np.array(left + list(reversed(right)), 'int32'), value)        


    def set_sure_area(self, matrix, value):
        self.set_sure_area_1(self.backbone, matrix, value, self.scaled_max_width, self.scaled_min_width)

    def set_max_area(self, backbone, matrix, value, min_radius, max_radius):
        max_val = -1
        min_val = -1
                
        for v in backbone:
            if v.value > max_val:
                max_val = v.value
            if min_val == -1 or v.value < min_val:
                min_val = v.value
     
        val_delta = max_val - min_val

        radii_delta = max_radius - min_radius
        
        for v in backbone:
            if val_delta != 0:
                r = min_radius + radii_delta * (v.value - min_val) / val_delta
            else:
                r = max_radius
            cv2.circle(matrix, (int(v.center.x), int(v.center.y)), int(r), value, -1)     


    def track(self, source, raw_matrix, weight_matrix, animals, frame_time):

        debug = []

        matrix = np.copy(raw_matrix)
        matrix = matrix.astype(float)        
        matrix1 = np.multiply(matrix, weight_matrix)
#       matrix1 = np.copy(matrix)
#       matrix = matrix.astype(np.uint8)

        debug_matrix = np.copy(matrix1)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug.append(("debug1 " + str(self.id), debug_matrix))
 
#        matrix = cv2.distanceTransform(matrix, cv2.DIST_L1, 5)

        matrix = cv2.filter2D(matrix1, -1, self.mass_center_filter)        
#        matrix = matrix.astype(np.float)
#        matrix = cv2.blur(matrix, (5, 5))
        
 #       matrix = np.multiply(matrix, matrix1)
                        
        debug_matrix = np.copy(matrix)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug.append(("gravity " + str(self.id), debug_matrix))
                
        bb1 = []        
        for v in self.backbone:
            bb1.append(v.clone())
                
#        self.backbone = self.boundary_aligner.align(matrix, weight_matrix, self.backbone, animals, frame_time)
        self.backbone = self.central_aligner.align(matrix, weight_matrix, self.backbone, animals, frame_time)
                
        return debug
                                
            
class TrackingFlowElement:
                
    def __init__(self, time, positions, filtered_image, debug):
        self.time = time
        self.positions = positions
        self.filtered_image = filtered_image  
        self.debug_frames = debug


class Tracking:
    
    class Configuration:
        skeletonization_res_width = 320
        skeletonization_res_height = 240        
        skeletonization_border = 20
        vertebra_length = 10
        scale = 1 

    scale_factor = 1        

    finished = False
    
    animals = []
    
    def __init__(self, video_file_name, config = Configuration(), logger = None):
        self.logger = logger
        self.config = config
        self.video = cv2.VideoCapture(video_file_name)
        frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.scale_factor = self.calculate_scale_factor(frame_width, frame_height)
        config.pixels_to_meters = float(config.scale) / frame_width
        config.max_animal_velocity = 1 # m/s
        config.vertebra_length = config.vertebra_length * self.scale_factor

    def calculate_scale_factor(self, frame_width, frame_height):
        width = self.config.skeletonization_res_width
        height = self.config.skeletonization_res_height
        k = float(frame_width) / frame_height
        if k > float(width) / height:
            return float(width) / frame_width
        else:
            return float(height) / frame_height
        
    def resize(self, frame):
        width = self.config.skeletonization_res_width
        height = self.config.skeletonization_res_height
        rows, cols = frame.shape[:2]        
        k = float(cols) / rows                
        if k > float(width) / height:
            cols = width
            rows = cols / k
        else:
            rows = height
            cols = rows * k
        frame = cv2.resize(frame, (int(cols), int(rows)))
        return frame    
        
    def calculate_background(self):

        ret, frame = self.video.read()
        rows, cols = frame.shape[:2]
        bg = np.zeros((rows, cols, 3), np.uint32)
        total = 0
        while True:        
            bg = bg + frame            
            total = total + 1
            ret, frame = self.video.read()
            if ret == False:
                break
            #if total > 10000:
#                break        
        bg = bg / total
        bg = bg.astype(np.uint8)        
        return bg
                

    def add_animal(self, start_x, start_y, end_x, end_y, config = Animal.Configuration()):
        self.animals.append(Animal(self, len(self.animals), start_x, start_y, end_x, end_y, config))
        return self.animals[-1]

    def track_animals(self, source, matrix_fs, matrix, frame_time):

        debug = []

        if len(self.animals) > 1:
            
            ret, thresh = cv2.threshold(matrix_fs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations = 3)
            debug.append(("otsu", thresh))

            thresh = thresh.astype(np.int32)        
            thresh[thresh == 255] = 1
                        
            rows, cols = source.shape[:2]        

            markers = np.copy(thresh)
        
        #    markers = np.ones((rows, cols), np.int32)

            index = 2        
            for a in self.animals:                                
                pos = a.get_position()                                
                a.set_sure_area_1(pos.backbone, markers, index, 15, 3)
                index += 1
        
            markers1 = np.copy(markers)        
        
            cv2.watershed(source, markers)

            cv2.normalize(markers1, markers1, 0, 255, cv2.NORM_MINMAX)           
            markers1 = markers1.astype(np.uint8)                

            source1 = np.copy(source)        
            source1[markers == -1] = (0, 0, 255)
 
            debug.append(("markers", markers1))
            debug.append(("segmented", source1))
        
            markers[markers == -1] = 1
            markers = markers.astype(np.uint8)
        
            markers = self.resize(markers)
            border = self.config.skeletonization_border
            markers = cv2.copyMakeBorder(markers, border, border, border, border, cv2.BORDER_CONSTANT, 1)

                
        weights = []
        rows, cols = matrix.shape[:2]

        index = 2
        for a in self.animals:
                                                                        
            weight = np.ones((rows, cols), np.float)                                    

            if len(self.animals) > 1:            

                max_occupied = np.zeros((rows, cols), np.uint8)                                                    

                for a1 in self.animals:
                    if a1 != a:
                        a1.set_max_area(a1.backbone, max_occupied, 255, a1.scaled_min_width, a1.scaled_max_width)

                
                debug.append(("ma" + str(index - 1), max_occupied))

                weight[(markers != index) & (markers != 1) & (max_occupied != 0)] = (0)
                #weight[(markers != index) & (markers != 1)] = (0)
            
            weights.append(weight)
            index += 1                    
        
        for a, w in zip(self.animals, weights):
                                                                                            
            debug1 = a.track(source, matrix, w, self.animals, frame_time)
            debug = debug + debug1
       
        return debug

        
    def track(self, bg, frame, frame_time):

        debug = []
                      
        frame_gr = cv2.absdiff(frame, bg)
        frame_gr = cv2.cvtColor(frame_gr, cv2.COLOR_BGR2GRAY)           
        cv2.normalize(frame_gr, frame_gr, 0, 255, cv2.NORM_MINMAX)           

        #kernel = np.ones((10, 10), np.uint8)
        #frame_gr = cv2.erode(frame_gr, kernel, iterations = 3)

        #frame_gr = morphology.skeletonize(frame_gr > 50)
        #frame_gr, distance = morphology.medial_axis(frame_gr > 50, return_distance = True)
        #frame_gr = bwmorph_thin.bwmorph_thin(frame_gr > 50, 30)
        #frame_gr = frame_gr.astype(np.uint8);

        #cv2.normalize(frame_gr, frame_gr, 0, 255, cv2.NORM_MINMAX)           

        frame_gr_resized = self.resize(frame_gr)           

        ret, frame_gr_resized = cv2.threshold(frame_gr_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame_gr_resized = morphology.remove_small_objects(frame_gr_resized > 0, 100, 2, in_place = True)        
        frame_gr_resized = frame_gr_resized.astype(np.uint8);
        cv2.normalize(frame_gr_resized, frame_gr_resized, 0, 255, cv2.NORM_MINMAX)           
        
        
        border = self.config.skeletonization_border   
        frame_gr_resized = cv2.copyMakeBorder(frame_gr_resized, border, border, border, border, cv2.BORDER_CONSTANT, 0)

                                             
        debug.append(("source", frame_gr))    

        debug1 = self.track_animals(frame, frame_gr, frame_gr_resized, frame_time)                   
        
        debug = debug + debug1
        
        positions = self.get_animal_positions()
        pos = positions[0][1]
        p = pos.backbone[pos.central_vertebra_index]        
        
        self.csv_path_writer.writerows([[p.center.x, p.center.y]])
        
        
        
                   
        tracking_flow_element = TrackingFlowElement(frame_time, positions, frame_gr, debug)       
                      
        return tracking_flow_element                        

    def get_animal_positions(self):
        positions = []        
        for a in self.animals:
            positions.append((a, a.get_position()))
        return positions

        
    def do_tracking(self, bg, start_frame, tracking_flow, time_to_stop, next_frame_semaphore, run_semaphore):

        pdb.set_trace()

        if not self.animals:
            return
            
        self.csv_path_file = open('path.csv', 'w')
        self.csv_path_writer = csv.writer(self.csv_path_file, delimiter=',')        
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)        
            
        # take current frame of the video
        ret, frame = self.video.read()

        if not ret:
            print('can\'t read the video')
            sys.exit()                               
            
        if not self.logger is None:
            self.logger.log("tracking started...")
            
        while(not time_to_stop.isSet()):
               
           if not run_semaphore.isSet():
               next_frame_semaphore.wait()
               next_frame_semaphore.clear()               

           frame_time = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.       
           
           e = self.track(bg, frame, frame_time)
                      
           # read the next frame
           ret, frame = self.video.read()    
           if ret == False:
               self.finished = True
                                 
           tracking_flow.put(e)
            
           if ret == False:
               break
         
        self.csv_path_file.close()
       
        if not self.logger is None:
            self.logger.log("tracking finished")
              
    
    