import sys
import numpy as np
import cv2
import math
import pdb
import time

from geometry import *
from vertebra import *
from scanner import *
            
class BoundaryAligner:
    
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
        
    def align_last_vertebra(self, matrix, v, start_v):
        
        flex_angle = math.pi * 30. / 180
        
        cos = geometry.pcosine(v.center, start_v.center, geometry.Point(start_v.center.x + 1, start_v.center.y))
        angle = math.acos(cos)
        if v.center.y > start_v.center.y:
            angle = 2*math.pi - angle
            #flexibility_angle = max_angle
    
        start_angle = angle - flex_angle / 2
        end_angle = angle + flex_angle / 2
        
        best_x = 0
        best_y = 0
        best_value = 0
    
        angle = start_angle
        
        dist = geometry.distance_p(start_v.center, v.center)
            
        while True:
            
            x = start_v.center.x + math.cos(angle) * dist
            y = start_v.center.y - math.sin(angle) * dist
    
            angle = angle + flex_angle / 10
        
            # bilinear interpolation
            x0 = int(math.floor(x))
            y0 = int(math.floor(y))
            x1 = int(math.ceil(x))
            y1 = int(math.ceil(y))
            
            value = float(matrix[y0, x0])*(x1 - x)*(y1 - y) + float(matrix[y0, x1])*(y1 - y)*(x - x0) + float(matrix[y1, x0])*(x1 - x)*(y - y0) + float(matrix[y1, x1])*(x - x0)*(y - y0)            
                    
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value
            
            if angle > end_angle:
                break
            
         # end scan cycle
            
        return (best_x, best_y, best_value)  
        
    def align_last_vertebra_limited(self, matrix, v, start_v, prev):
        
        flex_angle = math.pi * 30. / 180
        
        cos = geometry.pcosine(start_v.center, prev.center, geometry.Point(prev.center.x + 1, prev.center.y))
        angle = math.acos(cos)
        if start_v.center.y > prev.center.y:
            angle = 2*math.pi - angle
            
        start_angle = angle - flex_angle / 2
        end_angle = angle + flex_angle / 2
        
        best_x = 0
        best_y = 0
        best_value = 0
    
        angle = start_angle
        
        dist = geometry.distance_p(start_v.center, v.center)
            
        while True:
            
            x = start_v.center.x + math.cos(angle) * dist
            y = start_v.center.y - math.sin(angle) * dist
    
            angle = angle + flex_angle / 10
        
            # bilinear interpolation
            x0 = int(math.floor(x))
            y0 = int(math.floor(y))
            x1 = int(math.ceil(x))
            y1 = int(math.ceil(y))
            
            value = float(matrix[y0, x0])*(x1 - x)*(y1 - y) + float(matrix[y0, x1])*(y1 - y)*(x - x0) + float(matrix[y1, x0])*(x1 - x)*(y - y0) + float(matrix[y1, x1])*(x - x0)*(y - y0)            
                    
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value
            
            if angle > end_angle:
                break
            
         # end scan cycle
            
        return (best_x, best_y, best_value)  

        
    def align_middle_vertebra(self, matrix, prev, v, next):
        
        dx = next.center.x - prev.center.x
        dy = next.center.y - prev.center.y
        s = v.center
        e = geometry.Point(v.center.x + dx, v.center.y + dy)        
    
        dist = geometry.distance_p(prev.center, next.center)
            
        range = dist
                
        start_r = -range / 2.
        end_r = range / 2.
        
        best_x = 0
        best_y = 0
        best_value = 0
        best_inertia = 1
    
        r = start_r
         
        while True:
    
            p = geometry.point_along_a_perpendicular_p(s, e, v.center, r)
            
            r = r + range / 20.0
            
            x = p[0]
            y = p[1]
        
            # bilinear interpolation
            x0 = int(math.floor(x))
            y0 = int(math.floor(y))
            x1 = int(math.ceil(x))
            y1 = int(math.ceil(y))
            
            value = float(matrix[y0, x0])*(x1 - x)*(y1 - y) + float(matrix[y0, x1])*(y1 - y)*(x - x0) + float(matrix[y1, x0])*(x1 - x)*(y - y0) + float(matrix[y1, x1])*(x - x0)*(y - y0)            
                    
            inertia = 1 - abs(r) / (10*range)
            
            if value*inertia >= best_value:
                best_x = x
                best_y = y
                best_inertia = inertia
                best_value = value*inertia
            
            if r > end_r:
                break
            
         # end scan cycle
            
        return (best_x, best_y, best_value / best_inertia)  
                
    
    
    def align_backbone(self, matrix, weight_matrix, backbone, start_index, end_index, ref_value, min_vertebra, prev, min_value_coeff, frame_time):
        
        s = backbone[start_index]
        e = backbone[end_index]
        
        l = end_index - start_index + 1
                
        dist = geometry.distance_p(s.center, e.center)    
        if l < 3:
            if dist <= self.host.config.vertebra_length:
                return -1
            else:
                new_v = geometry.point_along_a_line_p(s.center, e.center, dist/2)
                index = start_index + 1
                backbone.insert(index, Vertebra(new_v.x, new_v.y))
                v = backbone[index]
        else:                
            
            if dist <= self.host.config.vertebra_length:
                del backbone[start_index + 1:end_index]
                return -1         
            
            index = start_index + l / 2
            v = backbone[index]
            
        (best_x, best_y, best_val) = self.align_middle_vertebra(matrix, s, v, e)                
        
        if best_val < ref_value*min_value_coeff:
            return index
                    
        v.center.x = best_x
        v.center.y = best_y
        v.value = best_val                
        
        r = self.align_backbone(matrix, weight_matrix, backbone, index, end_index, ref_value, min_vertebra, None, min_value_coeff, frame_time)
        if r != -1:
            return r
        r = self.align_backbone(matrix, weight_matrix, backbone, start_index, index, ref_value, min_vertebra, None, min_value_coeff, frame_time)
        if r != -1:
            return r
        return -1
            
    def align(self, matrix, weight_matrix, backbone, animals, frame_time):               
                                  
        max_i = 0
        max_val = -1
        for i in xrange(0, len(backbone) / 3):
            if backbone[i].value > max_val:
                max_val = backbone[i].value
                max_i = i
    
        #max_i = 0
                        
        central_vertebra_index = max_i
        
        cv = backbone[central_vertebra_index]
        (best_x, best_y, reference_value) = self.align_free_vertebra(matrix, backbone, cv)
        dx = best_x - cv.center.x
        dy = best_y - cv.center.y
        cv.value = reference_value
        
        # shift everything
        
        for v in backbone:
            v.center.x = v.center.x + dx
            v.center.y = v.center.y + dy        
    
        while True:
    
            while True:            
                (best_x, best_y, best_val) = self.align_last_vertebra(matrix, backbone[-1], backbone[max_i])        
                left = len(backbone) - max_i - 1
                if best_val < self.config.front_min_value_coeff * reference_value and left > 1:        
                    del backbone[-1]
                else:            
                    backbone[-1].center.x = best_x
                    backbone[-1].center.y = best_y
                    backbone[-1].value = best_val
                    break
                          
            e = len(backbone) - 1                                  
            
            r = self.align_backbone(matrix, weight_matrix, backbone, max_i, e, reference_value, 1, None, self.config.front_min_value_coeff, frame_time)
        
            if r != -1:
                self.host.logger.log("gap detected")
                backbone = backbone[0:r + 1]
            else:
                break            
            
        
        if max_i != 0:
            
            while True:                        
                (best_x, best_y, best_val) = self.align_last_vertebra_limited(matrix, backbone[0], backbone[max_i], backbone[max_i + 1])
                left = len(backbone) - max_i - 1
                if best_val < self.config.back_min_value_coeff * reference_value and left > 1:        
                    del backbone[0]
                else:            
                    backbone[0].center.x = best_x
                    backbone[0].center.y = best_y
                    backbone[0].value = best_val
                    break
            
            self.align_backbone(matrix, weight_matrix, backbone, 0, max_i, reference_value, 0, None, self.config.back_min_value_coeff, frame_time)            
     
     
        # try to extend the front        
        s = backbone[-2]
        e = backbone[-1]
        pvd = geometry.distance(s.center.x, s.center.y, e.center.x, e.center.y)
        next_center = geometry.point_along_a_line(s.center.x, s.center.y, e.center.x, e.center.y, pvd + self.host.config.vertebra_length)
        backbone.append(Vertebra(next_center[0], next_center[1], 0))
        (best_x, best_y, best_val) = self.align_last_vertebra(matrix, backbone[-1], backbone[-2])        
        if best_val < self.config.front_min_value_coeff * reference_value and left > 1:        
            del backbone[-1]
        else:            
            backbone[-1].center.x = best_x
            backbone[-1].center.y = best_y
            backbone[-1].value = best_val
            
        # try to extend the back
        s = backbone[1]
        e = backbone[0]
        pvd = geometry.distance(s.center.x, s.center.y, e.center.x, e.center.y)
        next_center = geometry.point_along_a_line(s.center.x, s.center.y, e.center.x, e.center.y, pvd + self.host.config.vertebra_length)
        backbone.insert(0, Vertebra(next_center[0], next_center[1], 0))
        (best_x, best_y, best_val) = self.align_last_vertebra(matrix, backbone[0], backbone[1])        
        if best_val < self.config.back_min_value_coeff * reference_value and left > 1:        
            del backbone[0]
        else:            
            backbone[0].center.x = best_x
            backbone[0].center.y = best_y
            backbone[0].value = best_val
        
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
            
            central_vertebra_index = max_i
        
        '''
        if float(len(backbone)) / self.max_vertebra < 0.4:
            backbone = self.backbone
        '''
        '''
        if float(len(backbone))  < 10:
            backbone = self.backbone
        '''

        self.last_frame_time = frame_time
            
        return (backbone, central_vertebra_index)
    
    