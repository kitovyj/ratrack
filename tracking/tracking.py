import sys
import numpy as np
import cv2
import math
import threading
import Queue
import pdb
import time

from geometry import *

class AnimalPosition:
    def __init__(self):
        self.backbone = []
        
class Animal:
        
    class Configuration:

        model_normal = 0
        model_with_drive = 1

        def __init__(self):
            self.model = self.model_normal
            self.max_body_length = 40
            self.front_min_value_coeff = 0.7
            self.back_min_value_coeff = 0.8
    
    class VertebraPosition:
        def __init__(self, x, y, value):
            self.center = geometry.Point(x, y)            
            self.value = value
            
    class Vertebra:
        def __init__(self, x, y, value):
            self.center = geometry.Point(x, y)            
            self.value = value
        def clone(self):
            return Animal.Vertebra(self.center.x, self.center.y, self.value)
                                    
    backbone = []            
        
    contours = None
    
    last_frame_time = None

    def __init__(self, host, id, start_x, start_y, end_x, end_y, config = Configuration()):
        
        self.host = host
        self.id = id
        self.config = config
        
        self.max_vertebra = config.max_body_length / self.host.config.vertebra_length
        
        hr = 6

        if config.model == self.config.model_with_drive:
            self.max_vertebra = (config.max_body_length * 1.3) / self.host.config.vertebra_length
        else:
            self.max_vertebra = config.max_body_length / self.host.config.vertebra_length
            
        head_radius = hr / host.scale_factor # 15
        front_radius = 7 / host.scale_factor # 17.5
        back_radius = 9 / host.scale_factor # 22.5
        mount_radius = 6 / host.scale_factor
        mount1_radius = 5 / host.scale_factor                                
                
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
            self.backbone.append(self.Vertebra(mount_position[0], mount_position[1], 0))            
            dist = dist + self.host.config.vertebra_length            
            
        self.central_vertebra_index = 1
            
        mass_center_filter_size = 20
        mass_center_filter_center = mass_center_filter_size / 2
        self.mass_center_filter = np.ones((mass_center_filter_size, mass_center_filter_size), np.float)
                
        for i in xrange(0, mass_center_filter_size):
            for j in xrange(0, mass_center_filter_size):
                if i != mass_center_filter_center or j != i:
                    dist = geometry.distance(i, j, mass_center_filter_center, mass_center_filter_center)
                    if dist < 10:
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
        self.set_sure_area_1(self.backbone, matrix, value, 7, 2)

                        
    class Scanner:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.vx = 1
            self.vy = 0
            self.r = 1
            self.s = 0
            
        def next(self):
            if self.s < self.r:
                self.s = self.s + 1;
                self.x = self.x + self.vx;
                self.y = self.y + self.vy;
                #print(self.x)                
                #print(self.y)
                return (self.x, self.y)
            else:
                if self.vx != 0:
                    self.vy = self.vx
                    self.vx = 0
                else:
                    self.r = self.r + 1
                    self.vx = -self.vy
                    self.vy = 0
                self.s = 0
                return self.next()        

    def align_free_vertebra(self, matrix, backbone, v, prev = 0, prev_prev = 0):
        
        scan_range_max = 5
        
        start_x = v.center.x
        start_y = v.center.y
           
        scanner = self.Scanner(start_x, start_y)
           
        best_value = 0
        best_x = start_x
        best_y = start_y

        first = True
           
        while True:               
            
            if not first:                       
                (x, y) = scanner.next()            
                border = self.host.config.skeletonization_border
                width = self.host.config.skeletonization_res_width
                height = self.host.config.skeletonization_res_height
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
        
        flexibility_angle = 2*math.pi / (len(self.backbone) - 1)
        
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
            k = 1
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
        
        max_vertebra_to_add = max(1, self.max_vertebra / 10)
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
               
               if best_value < min_value_coeff * ref_value:
                   if idx > min_vertebra:
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
                           
                           
               cos = geometry.cosine(best_x, best_y, prev.center.x, prev.center.y, v.center.x, v.center.y)
               angle = math.acos(cos)
               if best_y > prev.center.y:
                   angle = -angle                   
               self.host.logger.log('angle = ' + str(angle * 180 / math.pi))
               
               v.center.x = best_x
               v.center.y = best_y
               
               for v1 in backbone[idx + 1:]:
                   v1.center = geometry.Point.from_tuple(geometry.rotate_p(v1.center, v.center, angle))
               
               '''
               
               dx = best_x - v.center.x
               dy = best_y - v.center.y

               for v1 in backbone[idx:]:                                                  
                   v1.center.x = v1.center.x + dx
                   v1.center.y = v1.center.y + dy
               '''
                       
           # if it's the last vertebra, try to prolong the backbone...
           if idx == len(backbone) - 1 and vertebra_added < max_vertebra_to_add:             
              pvd = geometry.distance(prev.center.x, prev.center.y, v.center.x, v.center.y)
              next_center = geometry.point_along_a_line(prev.center.x, prev.center.y, v.center.x, v.center.y, pvd + self.host.config.vertebra_length)
              backbone.append(self.Vertebra(next_center[0], next_center[1], 0))
              vertebra_added = vertebra_added + 1
            
           idx = idx + 1

        return (backbone, central_value, central_index)

    def do_track(self, matrix, weight_matrix, backbone, animals, frame_time):               
                                  
        max_i = 0
        max_val = -1
        for i in xrange(0, len(self.backbone) / 3):
            if self.backbone[i].value > max_val:
                max_val = self.backbone[i].value
                max_i = i
        

        if max_i == 0:
            max_i = 1
                
        self.central_vertebra_index = max_i
    
                         
        cv = self.backbone[self.central_vertebra_index]
        (best_x, best_y, reference_value) = self.align_free_vertebra(matrix, backbone, cv)
        dx = best_x - cv.center.x
        dy = best_y - cv.center.y
        cv.value = reference_value
        
        # shift everything
        
        for v in self.backbone:
            v.center.x = v.center.x + dx
            v.center.y = v.center.y + dy
                            
        cvi = self.central_vertebra_index
                
        if cvi > 0:
            prev = self.backbone[cvi - 1]
        else:
            prev = 0
                            
        (new_front, front_val, front_index) = self.align_backbone(matrix, weight_matrix, self.backbone[cvi:], reference_value, 1, prev, self.config.front_min_value_coeff, frame_time)

        prev = new_front[1]

        (new_back, back_val, back_index) = self.align_backbone(matrix, weight_matrix, reversed(self.backbone[:cvi + 1]), reference_value, 0, prev, self.config.back_min_value_coeff, frame_time)

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

        if len(backbone) > self.max_vertebra: 
            
            bd = 10000;
            fd = 10000;
            
            for a in animals:
                if a == self:
                    continue
                for v in a.backbone:
                    d = geometry.pdistance(v.center, backbone[0].center)
                    bd = min(bd, d)
                    d = geometry.pdistance(v.center, backbone[-1].center)
                    fd = min(fd, d)
            
            if bd > fd:
                backbone = backbone[0:self.max_vertebra]
            else:
                backbone = backbone[len(backbone) - self.max_vertebra:]                
                                    
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
            
        return backbone
              
                        
    def track(self, source, raw_matrix, weight_matrix, animals, frame_time):

        debug = []

        matrix = np.copy(raw_matrix)
        matrix = matrix.astype(float)        
        matrix1 = np.multiply(matrix, weight_matrix)
#        matrix1 = np.copy(matrix)
#        matrix = matrix.astype(np.uint8)        

        debug_matrix = np.copy(matrix1)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug.append(("debug1 " + str(self.id), debug_matrix))
 
#      matrix = cv2.distanceTransform(matrix, cv2.DIST_L1, 5)

        matrix = cv2.filter2D(matrix1, -1, self.mass_center_filter)        
 #       matrix = np.multiply(matrix, matrix1)
                        
        debug_matrix = np.copy(matrix)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug.append(("center of mass " + str(self.id), debug_matrix))
                
        bb1 = []        
        for v in self.backbone:
            bb1.append(v.clone())
                
        self.backbone = self.do_track(matrix, weight_matrix, self.backbone, animals, frame_time)

        # find countour

        blurred = raw_matrix
                         

        rows, cols = raw_matrix.shape[:2]
        contour_mask = np.zeros((rows + 2, cols + 2), np.uint8)
        contour_mask.fill(1)
 
        u8weights = weight_matrix.astype(np.uint8)        
        others = cv2.copyMakeBorder(u8weights, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        others = 1 - others
        np.bitwise_or(contour_mask, others, contour_mask)                
        
        max_val = -1
        min_val = -1
                
        for v in self.backbone:
            if v.value > max_val:
                max_val = v.value
            if min_val == -1 or v.value < min_val:
                min_val = v.value
     
        val_delta = max_val - min_val

        start_radius = 12
        end_radius = 9
        radii_delta = start_radius - end_radius
        
        for v in self.backbone:
            if val_delta != 0:
                r = end_radius + radii_delta * (v.value - min_val) / val_delta
            else:
                r = start_radius
            cv2.circle(contour_mask, (int(v.center.x + 1), int(v.center.y + 1)), int(r), 0, -1)            
            
            
        ff_flags = (4 | 2 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        for v in self.backbone:
            val = blurred[int(v.center.y), int(v.center.x)]
            cv2.floodFill(blurred, contour_mask, (int(round(v.center.x)), int(round(v.center.y))), 2, 0.8 * val, 0.2 * val, flags = ff_flags)
#            cv2.floodFill(blurred, contour_mask, (v.center.x, v.center.y), 2, 0.5 * val, 1.2 * val, flags = ff_flags)
        ret, contour_mask = cv2.threshold(contour_mask, 1, 1, cv2.THRESH_BINARY_INV)
        
        self.contours = contour_mask
                
        debug_matrix = np.copy(contour_mask)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug_matrix = 255 - debug_matrix
        debug.append(("body contour " + str(self.id), debug_matrix))
        
        self.last_frame_time = frame_time
        
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
        vertebra_length = 6

    scale_factor = 1        

    finished = False
    
    animals = []
    
    def __init__(self, video_file_name, logger = None, config = Configuration()):
        self.logger = logger
        self.config = config
        self.video = cv2.VideoCapture(video_file_name)
        frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.scale_factor = self.calculate_scale_factor(frame_width, frame_height)

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
                

    def add_animal(self, start_x, start_y, end_x, end_y):
        self.animals.append(Animal(self, len(self.animals), start_x, start_y, end_x, end_y))
        return self.animals[-1]

    def track_animals(self, source, matrix_fs, matrix, frame_time):

        debug = []

        if len(self.animals) > 1:
            
            ret, thresh = cv2.threshold(matrix_fs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations = 3)
            debug.append(("otsu", thresh))

            thresh = thresh.astype(np.int32)        
            thresh[thresh == 255] = 1
                        
            rows, cols = source.shape[:2]        
            #markers = np.ones((rows, cols), np.int32)

            markers = np.copy(thresh)
        

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
            debug_weight = np.zeros((rows, cols), np.uint8)

            if len(self.animals) > 1:            
                weight[(markers != index) & (markers != 1)] = (0)
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
        frame_gr_resized = self.resize(frame_gr)           
        border = self.config.skeletonization_border   
        frame_gr_resized = cv2.copyMakeBorder(frame_gr_resized, border, border, border, border, cv2.BORDER_CONSTANT, 0)
                                             
        debug.append(("source", frame_gr))    

        debug1 = self.track_animals(frame, frame_gr, frame_gr_resized, frame_time)                   
        
        debug = debug + debug1
                   
        tracking_flow_element = TrackingFlowElement(frame_time, self.get_animal_positions(), frame_gr, debug)       
                      
        return tracking_flow_element                        

    def get_animal_positions(self):
        positions = []        
        for a in self.animals:
            positions.append((a, a.get_position()))
        return positions

        
    def do_tracking(self, bg, start_frame, tracking_flow, time_to_stop, next_frame_semaphore, run_semaphore):

#        pdb.set_trace()

        if not self.animals:
            return
        
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
       
        if not self.logger is None:
            self.logger.log("tracking finished")
              
    
    