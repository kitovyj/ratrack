import sys
import numpy as np
import cv2
import math
import pdb
import time

from geometry import *
from vertebra import *
from scanner import *

class CentralAligner:
    
    last_frame_time = None
    
    def __init__(self, animal):
        self.animal = animal
        self.host = animal.host
        self.config = animal.config
        
    def align_free_vertebra(self, matrix, backbone, v, prev = 0, prev_prev = 0):
        
        scan_range_max = 5
        
        start_x = v.center.x
        start_y = v.center.y
           
        scanner = Scanner(start_x, start_y)
           
        best_value = 0
        best_x = start_x
        best_y = start_y
    
        first = True
           
        while True:               
            
            if not first:                       
                (x, y) = scanner.next()            
                border = self.animal.host.config.skeletonization_border
                width = self.animal.host.config.skeletonization_res_width
                height = self.animal.host.config.skeletonization_res_height
                if x < border or y < border or x > border + width or y > border + height:
                    continue                
            else:
                (x, y) = (start_x, start_y)
                first = False
                   
            dist = geometry.distance(x, y, start_x, start_y)
               
            if dist > scan_range_max:
                break
               
            value = matrix[y, x]
                   
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value
           
         # end scan cycle
            
        return (best_x, best_y, best_value)            
        

    def align_vertebra(self, matrix, backbone, v, idx, prev, prev_prev, frame_time):
    
        if self.last_frame_time is None:
            time_passed = 1
        else:
            time_passed = frame_time - self.last_frame_time
            
        basic_max_rotation_velocity = math.pi # rad per second                
        #rvk = (idx * 10.0 + self.max_vertebra) / self.max_vertebra
        rvk = 1
        max_rotation_velocity = basic_max_rotation_velocity * rvk        
        max_angle = max_rotation_velocity * time_passed * 1000
    
        cos = geometry.pcosine(v.center, prev.center, geometry.Point(prev.center.x + 1, prev.center.y))
        own_angle = math.acos(cos)
        if v.center.y > prev.center.y:
            own_angle = 2*math.pi - own_angle
            
        start_rv_angle = own_angle - max_angle
        end_rv_angle = own_angle + max_angle                
        
        flexibility_angle = 2*math.pi / (len(self.animal.backbone) - 1)
        
        flexibility_angle = flexibility_angle * 0.5
                
        if prev_prev != 0:            
            cos = geometry.pcosine(prev.center, prev_prev.center, geometry.Point(prev_prev.center.x + 1, prev_prev.center.y))
            angle = math.acos(cos)
            if prev.center.y > prev_prev.center.y:
                angle = 2*math.pi - angle
        else:
            cos = geometry.pcosine(v.center, prev.center, geometry.Point(prev.center.x + 1, prev.center.y))
            angle = math.acos(cos)
            if v.center.y > prev.center.y:
                angle = 2*math.pi - angle
            #flexibility_angle = max_angle
    
        start_angle = angle - flexibility_angle / 2
        end_angle = angle + flexibility_angle / 2
    
    
        '''
        if not self.host.logger is None:
    
            if prev_prev != 0:
                self.host.logger.log('prev prev')
                self.host.logger.log(str(prev_prev.center.x) + ', ' + str(prev_prev.center.y))
            self.host.logger.log('prev')
            self.host.logger.log(str(prev.center.x) + ', ' + str(prev.center.y))
            self.host.logger.log('v')
            self.host.logger.log(str(v.center.x) + ', ' + str(v.center.y))
            self.host.logger.log('angles')
    
            self.host.logger.log(str(max_angle * 180 / math.pi))
            self.host.logger.log(str(own_angle * 180 / math.pi))
            self.host.logger.log(str(angle * 180 / math.pi))
            self.host.logger.log(str(start_rv_angle * 180 / math.pi))
            self.host.logger.log(str(end_rv_angle * 180 / math.pi))
            self.host.logger.log(str(start_angle * 180 / math.pi))
            self.host.logger.log(str(end_angle * 180 / math.pi))
        '''
            
        
    #            if not self.host.logger is None:
    #                self.host.logger.log("start_rv_angle > start_angle")
    
        '''        
        if start_rv_angle > start_angle:
            
            if start_rv_angle <= end_angle:
                start_angle = start_rv_angle
            else:
                start_angle = start_angle
                
        else:
            
           
                                    
                
        if end_rv_angle < end_angle:
    
            
            if not self.host.logger is None:
                self.host.logger.log("end_rv_angle < end_angle")
    #                self.host.logger.log(str(end_rv_angle * 180 / math.pi))
    #                self.host.logger.log(str(end_angle * 180 / math.pi))
           
            end_angle = max(end_rv_angle, start_angle)        
        '''
        
        angle = start_angle
    
        best_x = 0
        best_y = 0
        best_value = 0
        initialized = False
        best_k = 1
            
        while True:
            
            x = prev.center.x + math.cos(angle) * self.host.config.vertebra_length
            y = prev.center.y - math.sin(angle) * self.host.config.vertebra_length
    
            #angle = angle + (math.pi / 180) / 1
            angle = angle + flexibility_angle / 10
            
            
            cos = geometry.pcosine(v.center, prev.center, geometry.Point(x, y))            
            rot = math.acos(cos)
            
            '''
            rotation_probability = [ 1.0, 0.9, 1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 
                                     0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ]
            '''
            
            rk = rot / flexibility_angle
                                                         
            #rotation_inertia = rotation_probability[int(min(round(rot / (math.pi / 18)), 17))]            
            
            #rotation_inertia = 1.0
            
            #k =  1 - 0.0001 * rot * (float(self.max_vertebra) / len(self.backbone)**3)
            k =  1 - 0.01 * rot
            #k = 1
            #k = 1
    #            k = 1
    
            '''
            if rot > max_angle:
                if (angle < end_angle) or initialized:
                    continue
                else:
    
                    x0 = prev.center.x + math.cos(start_angle) * self.host.config.vertebra_length
                    y0 = prev.center.y - math.sin(start_angle) * self.host.config.vertebra_length
                    x1 = prev.center.x + math.cos(end_angle) * self.host.config.vertebra_length
                    y1 = prev.center.y - math.sin(end_angle) * self.host.config.vertebra_length
    
                    cos = geometry.pcosine(v.center, prev.center, geometry.Point(x0, y0))            
                    rot1 = math.acos(cos)
                    cos = geometry.pcosine(v.center, prev.center, geometry.Point(x1, y1))            
                    rot2 = math.acos(cos)
                    
                    if rot1 < rot2:
                        x = x0
                        y = y0
                    else:
                        x = x1
                        y = y1
                                        
                    self.host.logger.log('max angle reached...')                                        
    
            initialized = True
            '''
        
            # bilinear interpolation
            x0 = int(math.floor(x))
            y0 = int(math.floor(y))
            x1 = int(math.ceil(x))
            y1 = int(math.ceil(y))
            
            value = float(matrix[y0, x0])*(x1 - x)*(y1 - y) + float(matrix[y0, x1])*(y1 - y)*(x - x0) + float(matrix[y1, x0])*(x1 - x)*(y - y0) + float(matrix[y1, x1])*(x - x0)*(y - y0)            
                    
            if k*value >= best_value:
                best_x = x
                best_y = y
                best_value = value*k
                best_k = k
            
            if angle > end_angle:
                break
            
         # end scan cycle
            
        return (best_x, best_y, best_value / best_k)  
        
    def align_backbone(self, matrix, weight_matrix, original_backbone, ref_value, min_vertebra, prev, min_value_coeff, frame_time):
    
        central_value = 0
        central_index = 0
        
        max_vertebra_to_add = max(1, self.animal.max_vertebra / 10)
        vertebra_added = 0
    
        backbone = []        
        for v in original_backbone:
            backbone.append(v.clone())
        
        idx = 0
        while idx < len(backbone):
                           
           v = backbone[idx]
    
           v.value = matrix[int(v.center.y), int(v.center.x)]
                               
           if idx > 1:               
               prev_prev = backbone[idx - 2]
           else:
               if idx == 1:
                   prev_prev = prev
               else:
                   prev_prev = 0
    
           if idx > 0:               
               prev = backbone[idx - 1]
               
           if idx > 0:
               
               (best_x, best_y, best_value) = self.align_vertebra(matrix, backbone, v, idx, prev, prev_prev, frame_time)
               
               #self.host.logger.log(str(best_value))
    
               if best_value < min_value_coeff * ref_value:
                   #self.host.logger.log(str(best_value))
                   #self.host.logger.log(str(idx) + " " + str(len(backbone)))
    
                   if idx > min_vertebra and idx == len(backbone) - 1:
                       backbone = backbone[:idx]
                       break
                               
               '''                                
               dist = geometry.distance(prev.center.x, prev.center.y, best_x, best_y)
               do_break = False
               for l in range(1, int(dist)):
                   p = geometry.point_along_a_line(prev.center.x, prev.center.y, best_x, best_y, l)
                   val = matrix[p[1], p[0]]
                   min_val = min(best_value, prev.value)
                   #diff = abs(best_value - prev.value)
                   if val < 0.95*min_val:
                       #and abs(val - min_val) > 4*diff:
                       if idx > min_vertebra:
                           backbone = backbone[:idx]
                       do_break = True
                       break
                   
               if do_break:
                   break
               '''
               
               v.value = best_value
           
               if best_value > central_value:
                   central_value = best_value
                   central_index = idx
                           
                           
               angle = - geometry.angle(best_x, best_y, prev.center.x, prev.center.y, v.center.x, v.center.y)
               
               v.center.x = best_x
               v.center.y = best_y
               
               first = True
               for v1 in backbone[idx + 1:]:
                   v1.center = geometry.Point.from_tuple(geometry.rotate_p(v1.center, v.center, angle))
                                                         
           # if it's the last vertebra, try to prolong the backbone...
           
           if idx == len(backbone) - 1 and vertebra_added < max_vertebra_to_add:             
              pvd = geometry.distance(prev.center.x, prev.center.y, v.center.x, v.center.y)
              next_center = geometry.point_along_a_line(prev.center.x, prev.center.y, v.center.x, v.center.y, pvd + self.host.config.vertebra_length)
              backbone.append(Vertebra(next_center[0], next_center[1], 0))
              vertebra_added = vertebra_added + 1
           
            
           idx = idx + 1
    
        return (backbone, central_value, central_index)
    
    
    def align(self, matrix, weight_matrix, backbone, animals, frame_time):               
                                  
        max_i = 0
        max_val = -1
        for i in xrange(0, len(backbone) / 3):
            if backbone[i].value > max_val:
                max_val = backbone[i].value
                max_i = i
        
    
        '''
        if max_i == 0:
            max_i = 1
        '''
                
        self.central_vertebra_index = max_i
    
                         
        cv = backbone[self.central_vertebra_index]
        (best_x, best_y, reference_value) = self.align_free_vertebra(matrix, backbone, cv)
        dx = best_x - cv.center.x
        dy = best_y - cv.center.y
        cv.value = reference_value
        
        # shift everything
        
        for v in backbone:
            v.center.x = v.center.x + dx
            v.center.y = v.center.y + dy
                            
        cvi = self.central_vertebra_index
                
        if cvi > 0:
            prev = backbone[cvi - 1]
        else:
            prev = 0
                            
        (new_front, front_val, front_index) = self.align_backbone(matrix, weight_matrix, backbone[cvi:], reference_value, 1, prev, self.config.front_min_value_coeff, frame_time)
    
        prev = new_front[1]
    
        (new_back, back_val, back_index) = self.align_backbone(matrix, weight_matrix, reversed(backbone[:cvi + 1]), reference_value, 0, prev, self.config.back_min_value_coeff, frame_time)
    
        self.central_vertebra_index = len(new_back) - 1
    
        if back_val > reference_value:
            reference_value = back_val
            self.central_vertebra_index = len(new_back) - back_index - 1 
        
        if front_val > reference_value:
            reference_value = front_val
            self.central_vertebra_index = len(new_back) + front_index - 1
                        
                
        backbone = list(reversed(new_back)) + new_front[1:]
    
        if self.central_vertebra_index == len(backbone) - 1:
            self.central_vertebra_index = len(backbone) - 2
    
        if len(backbone) > self.animal.max_vertebra: 
            
            bd = 10000;
            fd = 10000;
            
            for a in animals:
                if a == self:
                    continue
                for v in a.backbone:
                    d = geometry.distance_p(v.center, backbone[0].center)
                    bd = min(bd, d)
                    d = geometry.distance_p(v.center, backbone[-1].center)
                    fd = min(fd, d)
            
            if fd < bd:
                backbone = backbone[0:self.animal.max_vertebra]
            else:
                backbone = backbone[-self.animal.max_vertebra:]                
                                    
            '''
            best_i = 0
            best_sum = 0                                
            
            i = 0
            for i in range(0, len(backbone) - self.max_vertebra - 1):
                sum = 0
                for j in range(0, self.max_vertebra):
                    sum = sum + backbone[j].value
                if sum > best_sum:
                    best_sum = sum
                    best_i = i
            
            backbone = backbone[best_i:best_i + self.max_vertebra]
            '''
            
            max_val = 0
            max_i = 0
            for idx, v in enumerate(backbone):
                if v.value > max_val:
                    max_val = v.value
                    max_i = idx
                    
            if max_i == len(backbone) - 1:
                max_i = max_i - 1
            
            self.central_vertebra_index = max_i
        
        '''
        if float(len(backbone)) / self.max_vertebra < 0.4:
            backbone = self.backbone
        '''
        '''
        if float(len(backbone))  < 10:
            backbone = self.backbone
        '''
        self.last_frame_time = frame_time
            
        return backbone
              
                        
    