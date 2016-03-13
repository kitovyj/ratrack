import sys
import numpy as np
import cv2
import math
import threading
import Queue
import pdb
import geometry
import time

from geometry import Point

tracking_border = 20
tracking_resolution_width = 320
tracking_resolution_height = 240

curr_cos = 0

max_animal_length = 40
max_animal_with_drive_length = 55

# histogram usage: http://opencvpython.blogspot.nl/2013/03/histograms-4-back-projection.html

def resize(frame):
    width = tracking_resolution_width
    height = tracking_resolution_height
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

class TrackingParams:
    
    scale_factor = 1        
    best_fit_search_radius = 1;
    tracking_border = 20
    
    def __init__(self, scale_factor = 1):
        self.scale_factor = scale_factor


class BodyPart:
            
    value = 0.; # cover value
    vx = 0.;
    vy = 0.;
    dx = 0.;
    dy = 0.;
    area = 0.;
    triangle = False;
    prev = 0.;
    mean_value = 0.;
        
    def __init__(self, params, x, y, radius, prev = 0., triangle = False):
        self.params = params
        self.center = Point(x * params.scale_factor + tracking_border, 
                            y * params.scale_factor + tracking_border)
        self.original_radius = radius
        self.radius = radius * params.scale_factor
        if not triangle:
            self.area = math.pi*(self.radius**2)
        else:
            self.area = (3. / 4) * math.sqrt(3) * (self.radius**2)
            
        self.triangle = triangle
        self.prev = prev        
        
        extra_space = 5
        mass_center_filter_size = int(self.radius * 2 + 1 + extra_space)
        mass_center_filter_center = (mass_center_filter_size + 1) / 2
        self.mass_center_filter = np.ones((mass_center_filter_size, mass_center_filter_size), np.float)
        
        for i in xrange(0, mass_center_filter_size):
            for j in xrange(0, mass_center_filter_size):
                dist = geometry.distance(i, j, mass_center_filter_center, mass_center_filter_center)
                self.mass_center_filter[i, j] = (self.radius - dist) / self.radius
        

    def set_weights(self, matrix, weight):        
        if not self.triangle:
            cv2.circle(matrix, (int(self.center.x), int(self.center.y)), int(self.radius), weight, -1)
        else:
            hc = self.center
            fc = self.prev.center
            hr = self.radius
            fhd = geometry.distance(fc.x, fc.y, hc.x, hc.y)        

            side = math.sqrt(3.) * hr
            height = 3. * hr / 2.

            top = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd + hr)
            bottom = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd - height / 2)
          
            left = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                        bottom[0], bottom[1], side / 2)
            right = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                         bottom[0], bottom[1], -side / 2)
                                                         
            cv2.fillConvexPoly(matrix, np.array([list(top), list(left), list(right)], 'int32'), weight)
        
        
    def shift(self, matrix):        
        sum_x = 0
        sum_y = 0
        values_sum = 0
        total_elements = 0
        for x in xrange(self.center.x - self.radius, self.center.x + self.radius):
            for y in xrange(self.center.y - self.radius, self.center.y + self.radius):
                distance_from_center = math.sqrt((self.center.x - x)**2 + (self.center.y - y)**2)
                if distance_from_center < self.radius:
                    total_elements += 1
                    value = matrix[y, x]
                    values_sum += value
                    sum_x += x * value
                    sum_y += y * value
        if values_sum != 0:
            center.x = sum_x / values_sum;
            center.y = sum_y / values_sum;
            
    def belongs_to(self, given_center_x, given_center_y, x, y):
        distance_from_center = math.sqrt((given_center_x - x)**2 + (given_center_y - y)**2)
        return distance_from_center < self.radius
        
    # todo: add mitex synchronization to preperties retrieval methods
        
    def get_position(self):
        return Point((self.center.x - tracking_border) / self.params.scale_factor, 
                     (self.center.y - tracking_border) / self.params.scale_factor)

    def get_radius(self):
        return self.original_radius
        
    def get_inner_radius(self):
        if self.triangle:            
            return self.radius / 2
        else:
            return self.radius
        
                
animal_n = 0

class AnimalPosition:
    def __init__(self):
        self.head = 0
        self.front = 0
        self.back = 0
        self.mount = 0
        self.mount1 = 0

class Animal:
    
    class PartConfiguration:
        def __init__(self, x, y, part, val):
            self.part = part
            self.x = x
            self.y = y
            self.value = val # cover value

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
    
    vertebra_dist = 2
    
    contours = None

    # deduce body parts positions from back-to-front vector
    def __init__(self, params, start_x, start_y, end_x, end_y):

        global animal_n
        
        self.params = params        
        self.max_vertebra = max_animal_length / self.vertebra_dist
        
        self.mount_visible = True
        self.mount1_visible = True
        self.animal_number = animal_n + 1

        hr = 6

        self.front_min_value_coeff = 0.5
        self.back_min_value_coeff = 0.8

        if animal_n >= 1:
            self.front_min_value_coeff = 0.7
            self.max_vertebra = max_animal_length / self.vertebra_dist
        else:
            self.front_min_value_coeff = 0.6
            self.max_vertebra = max_animal_with_drive_length / self.vertebra_dist
            

        head_radius = hr / params.scale_factor # 15
        front_radius = 7 / params.scale_factor # 17.5
        back_radius = 9 / params.scale_factor # 22.5
        mount_radius = 6 / params.scale_factor
        mount1_radius = 5 / params.scale_factor

        
#        head_radius = 4 / params.scale_factor
#        front_radius = 5 / params.scale_factor
#        back_radius = 7 / params.scale_factor
#        mount_radius = 6 / params.scale_factor

#        head_radius = 3
#        front_radius = 5
#        back_radius = 7
                
        length = math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
        
                
        total = 2*back_radius + 2*front_radius + 2*head_radius
        
        if animal_n == 0:
            total = total + 2*mount_radius + 2*mount1_radius

        back_position = geometry.point_along_a_line(end_x, end_y, start_x, 
                                                    start_y, length * float(back_radius) / total)
        front_position = geometry.point_along_a_line(end_x, end_y, 
                                                     start_x, start_y, length * float(2*back_radius + front_radius) / total)
        head_position = geometry.point_along_a_line(end_x, end_y, 
                                                    start_x, start_y, length * float(2*back_radius + 2*front_radius + head_radius) / total)

        mount_position = geometry.point_along_a_line(end_x, end_y, 
                                                     start_x, start_y, length * float(2*back_radius + 2*front_radius + 2*head_radius + mount_radius) / total)

        mount1_position = geometry.point_along_a_line(end_x, end_y, 
                                                      start_x, start_y, length * float(2*back_radius + 2*front_radius + 2*head_radius + 2*mount_radius + mount1_radius) / total)


        if animal_n == 0:
            self.mount = BodyPart(params, int(mount_position[0]), int(mount_position[1]), mount_radius)
            self.mount1 = BodyPart(params, int(mount1_position[0]), int(mount1_position[1]), mount1_radius)
        else:
            self.mount = 0
            self.mount1 = 0

        self.back = BodyPart(params, int(back_position[0]), int(back_position[1]), back_radius)
        self.front = BodyPart(params, int(front_position[0]), int(front_position[1]), front_radius)
        self.head = BodyPart(params, int(head_position[0]), int(head_position[1]), head_radius, self.front)
#        self.head = BodyPart(params, int(head_position[0]), int(head_position[1]), head_radius, self.front, self.mount == 0)
        
        start_x =  start_x * params.scale_factor + tracking_border
        start_y =  start_y * params.scale_factor + tracking_border
        end_x =  end_x * params.scale_factor + tracking_border
        end_y =  end_y * params.scale_factor + tracking_border
                
        max_dist = geometry.distance(end_x, end_y, start_x, start_y)
        
        dist = 0
        
        if max_dist < self.vertebra_dist * 2:
            max_dist = self.vertebra_dist * 2

        print(len(self.backbone))
        
        self.backbone = []
        
        while dist <= max_dist:            
            mount_position = geometry.point_along_a_line(end_x, end_y, start_x, start_y, dist)            
            self.backbone.append(self.Vertebra(mount_position[0], mount_position[1], 0))            
            dist = dist + self.vertebra_dist            
            
        self.central_vertebra_index = 1
            
        animal_n = animal_n + 1;
        
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
        for v in self.backbone:
            r.backbone.append(self.VertebraPosition((v.center.x - tracking_border) / self.params.scale_factor, 
                                                    (v.center.y - tracking_border) / self.params.scale_factor, v.value))
        r.central_vertebra_index = self.central_vertebra_index
        r.head = self.head.get_position()
        r.front = self.front.get_position()
        r.back = self.back.get_position()
        if self.mount != 0:
            r.mount = self.mount.get_position()
        if self.mount1 != 0:
            r.mount1 = self.mount1.get_position()
            
        return r
                
        '''
        r = AnimalPosition()
        r.head = self.head.get_position()
        r.front = self.front.get_position()
        r.back = self.back.get_position()
        if self.mount != 0:
            r.mount = self.mount.get_position()
        if self.mount1 != 0:
            r.mount1 = self.mount1.get_position()
        return r
        '''
                                    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)        

    def set_weights_no_mount(self, matrix, weight):        
        
        
        if not (self.contours is None):
            rows, cols = self.contours.shape[:2]            
            np.multiply(matrix, self.contours[1:rows - 1, 1:cols - 1], matrix)
         

        '''
        max_val = -1
        min_val = -1
                
        for v in self.backbone:
            if v.value > max_val:
                max_val = v.value
            if min_val == -1 or v.value < min_val:
                min_val = v.value
     
        val_delta = max_val - min_val

        start_radius = 12 + 2
        end_radius = 9 + 2
        radii_delta = start_radius - end_radius
        step = float(start_radius - end_radius) / len(self.backbone)
        r = start_radius

        for v in self.backbone:
            if val_delta != 0:
                r = end_radius + radii_delta * (v.value - min_val) / val_delta
            else:
                r = start_radius
            cv2.circle(matrix, (int(v.center.x), int(v.center.y)), int(r), weight, -1)            
#            r = r - step
        
        '''
        '''
        self.front.set_weights(matrix, weight)        
        self.back.set_weights(matrix, weight)
        self.head.set_weights(matrix, weight)
        
        hc = self.head.center
        fc = self.front.center
        bc = self.back.center
        hr = self.head.radius
        fr = self.front.radius
        br = self.back.radius

        if self.head.triangle:
        
            head_side = math.sqrt(3.) * hr
            head_height = 3. * hr / 2.

            fhd = geometry.distance(fc.x, fc.y, hc.x, hc.y)                
            head_bottom = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd - head_height / 2)
                    
            head_p1 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                           head_bottom[0], head_bottom[1], head_side / 2)
            head_p2 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                           head_bottom[0], head_bottom[1], -head_side/2)
                                                           
        else:
            
            head_p1 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                           hc.x, hc.y, hr)
            head_p2 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                           hc.x, hc.y, -hr)
        
        front_p1 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                        fc.x, fc.y, fr)
        front_p2 = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y,                                                                                                                 
                                                        fc.x, fc.y, -fr)
                                                        
        cv2.fillConvexPoly(matrix, np.array([list(head_p1), list(front_p1), list(front_p2), list(head_p2)], 'int32'), weight)
                                                                
        front_p1 = geometry.point_along_a_perpendicular(fc.x, fc.y, bc.x, bc.y, 
                                                        fc.x, fc.y, fr)
        front_p2 = geometry.point_along_a_perpendicular(fc.x, fc.y, bc.x, bc.y, 
                                                        fc.x, fc.y, -fr)
        back_p1 = geometry.point_along_a_perpendicular(fc.x, fc.y, bc.x, bc.y, 
                                                       bc.x, bc.y, br)
        back_p2 = geometry.point_along_a_perpendicular(fc.x, fc.y, bc.x, bc.y, 
                                                       bc.x, bc.y, -br)
        cv2.fillConvexPoly(matrix, np.array([list(back_p1), list(front_p1), list(front_p2), list(back_p2)], 'int32'), weight)
        '''
        

    def set_weights(self, matrix, weight, mount_weight):
        self.set_weights_no_mount(matrix, weight)                
        '''
        if self.mount != 0 and self.mount_visible:
            self.mount.set_weights(matrix, mount_weight)
        if self.mount1 != 0 and self.mount1_visible:
            self.mount1.set_weights(matrix, mount_weight)
        '''
#        self.set_weights_no_mount(matrix, weight)
        
    def mean_value(self, matrix, bp):
        radius = bp.radius
        center_x = bp.center.x
        center_y = bp.center.y
        area = math.pi * (radius**2)
        mask = np.zeros((int(radius*2) + 1, int(radius*2) + 1), np.float)        
        cv2.circle(mask, (int(radius), int(radius)), int(radius), (1.0), -1)                    
        masked = np.multiply(mask, matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])
        return masked.sum() / area
                
    def weight(self, matrix, center_x, center_y, radius, weight_matrix):

        mask = np.zeros((int(radius*2) + 1, int(radius*2) + 1), np.float)        
        cv2.circle(mask, (int(radius), int(radius)), int(radius), (1.0), -1)                    
        weighted_mask = np.multiply(mask, weight_matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])
        weighted =  np.multiply(weighted_mask, matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])        
        return (weighted.sum(), weighted_mask.sum());

    def triangle_weight(self, matrix, center_x, center_y, radius, org_x, org_y, weight_matrix, added_height = 0.):
                 
        mask = np.zeros((int((radius + added_height)*2) + 1, int((radius + added_height)*2) + 1), np.float)        
                        
        hc = Point(center_x, center_y)            
        fc = Point(org_x, org_y)
        hr = radius
        fhd = geometry.distance(fc.x, fc.y, hc.x, hc.y)        

        side = math.sqrt(3.) * hr
        height = 3. * hr / 2.

        top = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd + hr + added_height)
        bottom = geometry.point_along_a_line(fc.x, fc.y, hc.x, hc.y, fhd - height / 2)
          
        left = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                    bottom[0], bottom[1], side / 2 + added_height)
        right = geometry.point_along_a_perpendicular(fc.x, fc.y, hc.x, hc.y, 
                                                     bottom[0], bottom[1], - side / 2 - added_height)
                                                                     
        ref = (- center_x + radius + added_height, - center_y + radius + added_height)

        top = tuple(np.add(top, ref))
        left = tuple(np.add(left, ref))
        right = tuple(np.add(right, ref))
            
        cv2.fillConvexPoly(mask, np.array([list(top), list(left), list(right)], 'int32'), 1.0)
                                                               
        radius = radius + added_height                                                               
        
        weighted_mask = np.multiply(mask, weight_matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])
        weighted =  np.multiply(weighted_mask, matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])        

        return (weighted.sum(), weighted_mask.sum());


    def bfcover(self, matrix, center_x, center_y, part, weight_matrix):        
        inner = self.weight(matrix, center_x, center_y, part.radius, weight_matrix)
        return inner[0]

    def cover(self, matrix, center_x, center_y, part, weight_matrix, subtract_surroundings = False):
        if not part.triangle:
            inner = self.weight(matrix, center_x, center_y, part.radius, weight_matrix)
            if not subtract_surroundings:        
                if inner[1] == 0:
                    return (0.0, inner[1])
                return (inner[0] / inner[1], inner[1])
            outer_r = part.radius + 4
            outer = self.weight(matrix, center_x, center_y, outer_r, weight_matrix)        
            val = (inner[0] - 0.5*(outer[0] - inner[0]))
            if inner[1] == 0:
                return (0.0, inner[1])
            return (val / inner[1], inner[1])
        else:
            inner = self.triangle_weight(matrix, center_x, center_y, part.radius, part.prev.center.x, part.prev.center.y, weight_matrix)
            outer = self.triangle_weight(matrix, center_x, center_y, part.radius, part.prev.center.x, part.prev.center.y, weight_matrix, 4)
            if not subtract_surroundings:        
                if inner[1] == 0:
                    return (0.0, inner[1])
                return (inner[0] / inner[1], inner[1])                
            val = (inner[0] - 0.5*(outer[0] - inner[0]))
            if inner[1] == 0:
                return (0.0, inner[1])
            return (val / inner[1], inner[1])
        
    # recursive brute force "best fit"
    def do_fit(self, matrix, parts, configuration, weight_matrix):

        max = 0
        result_config = list(configuration)
        search_radius = self.params.best_fit_search_radius
        part, rest = parts[0], parts[1:]
        for x in xrange(int(part.center.x) - search_radius, int(part.center.x) + search_radius):
            for y in xrange(int(part.center.y) - search_radius, int(part.center.y) + search_radius):

                if configuration:
                    dst = math.sqrt((configuration[-1].x - x)**2 + (configuration[-1].y - y)**2)
                    if dst - (configuration[-1].part.radius + part.radius) > 1.5:
                        continue
                    
                if len(configuration) > 1:
                    #calculate the angle
                    c1 = configuration[-2]
                    c2 = configuration[-1]
                    x1 = c1.x - c2.x
                    y1 = c1.y - c2.y
                    x2 = x - c2.x
                    y2 = y - c2.y
                    dot = x1 * x2 + y1 * y2
                    a = math.sqrt(x1**2 + y1**2)
                    b = math.sqrt(x2**2 + y2**2)
                    if a * b != 0:
                        cos = dot / (a * b)
                        if cos > 0.1:
                            continue
                               
                cover_value = self.bfcover(matrix, x, y, part, weight_matrix);                

                new_config = list(configuration)
                new_config.append(self.PartConfiguration(x, y, part, cover_value))                

                local_max = 0
                if rest:
                    new_weight_matrix = np.copy(weight_matrix)
                    cv2.circle(new_weight_matrix, (x, y), int(part.radius), (0.7), -1)                    
                    (local_max, local_max_config) = self.do_fit(matrix, rest, new_config, new_weight_matrix)
                else:
                    local_max_config = new_config
                
                local_max += cover_value;
                if local_max > max:
                    max = local_max
                    result_config = local_max_config
                    
        return (max, result_config)
                
                
    
    def best_fit(self, matrix):

        border = tracking_border   
        frame = resize(matrix)
        frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, (0, 0, 0))  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = (255 - frame)
        
        parts = [self.back, self.front, self.head]
        
        if self.mount:
            parts.append(self.mount)

        rows, cols = matrix.shape[:2]

        weight_matrix = np.ones((rows, cols), np.float)

        (max, configuration) = self.do_fit(frame, parts, [], weight_matrix)
        
        for part_config in configuration:
            part_config.part.center.x = part_config.x
            part_config.part.center.y = part_config.y
            part_config.part.value = part_config.value
                        
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

        max_angle = min(math.pi / len(self.backbone), math.pi / 6)
        flexibility_angle = math.pi - max_angle
        
        scan_range_min = self.vertebra_dist - 1
        scan_range_max = self.vertebra_dist + 1        
        
        if prev == 0:               
            start_x = v.center.x
            start_y = v.center.y
            scan_range_min = 0
            scan_range_max = self.vertebra_dist + 1
        else:
            start_x = prev.center.x
            start_y = prev.center.y
            scan_range_min = self.vertebra_dist - 1
            scan_range_max = self.vertebra_dist + 1
           
        scanner = self.Scanner(int(start_x), int(start_y))
           
        best_value = 0
        best_x = start_x
        best_y = start_y

        first = True
           
        while True:               
            
            if not first:                       
                (x, y) = scanner.next()            
                if x < tracking_border or y < tracking_border or x > tracking_border + tracking_resolution_width or y > tracking_border + tracking_resolution_height:
                    continue                
            else:
                (x, y) = (start_x, start_y)
                first = False
                   
            dist = geometry.distance(x, y, start_x, start_y)

            if dist > scan_range_max + 2:
                break
               
            if dist < scan_range_min or dist > scan_range_max:
                continue


            rotation_inertia = 1.

            # flexibility
            if prev_prev != 0:
                cosine = geometry.cosine(prev_prev.center.x, prev_prev.center.y, prev.center.x, prev.center.y, x, y)          
                angle = math.acos(cosine)
                if angle < flexibility_angle:
                    continue
               
            if prev != 0:

                cosine = geometry.cosine(v.center.x, v.center.y, prev.center.x, prev.center.y, x, y)          
                angle = math.acos(cosine)
                if angle > max_angle:
                    continue
                   
                rotation_probability = [ 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 
                                         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ]
                                                         
                rotation_inertia = rotation_probability[int(min(round(angle / (math.pi / 18)), 17))]
                                                                           
            stretch_inertia = 1.
            motion_inertia = 1.
               
            if prev != 0:
                stretch_probability = [ 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5 ]
                was = geometry.distance(v.center.x, v.center.y, prev.center.x, prev.center.y) 
                now = geometry.distance(x, y, prev.center.x, prev.center.y) 
                stretch = abs(was - now)
                stretch_inertia = stretch_probability[int(min(stretch, 9))]
            else:
                motion_probability = [ 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5 ]                           
                motion = geometry.distance(v.center.x, v.center.y, x, y) 
                motion_inertia = motion_probability[int(min(motion, 9))]


            inertia = rotation_inertia * stretch_inertia * motion_inertia    
               
            inertia = 1.
            #inertia = math.pow(inertia, 1./6)
               
            value = matrix[y, x] * inertia
                   
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value
           
         # end scan cycle
            
        return (best_x, best_y, best_value)            

    def align_vertebra(self, matrix, backbone, v, prev = 0, prev_prev = 0):

        flexibility_angle = 2*math.pi / (len(self.backbone) - 1)
        
        flexibility_angle = flexibility_angle / 2
                
        if prev_prev != 0:            
            cos = geometry.pcosine(prev.center, prev_prev.center, geometry.Point(prev_prev.center.x + 1, prev_prev.center.y))
            angle = math.acos(cos)
            if prev.center.y < prev_prev.center.y:
                angle = 2*math.pi - angle
        else:
            cos = geometry.pcosine(v.center, prev.center, geometry.Point(v.center.x + 1, v.center.y))
            angle = math.acos(cos)
            if v.center.y < prev.center.y:
                angle = 2*math.pi - angle

        start_angle = angle - flexibility_angle / 2
        end_angle = angle + flexibility_angle / 2
        
        angle = start_angle

        best_value = 0
        best_x = 0
        best_y = 0
            
        while True:
            
            x = prev.center.x + math.cos(angle) * self.vertebra_dist
            y = prev.center.y + math.sin(angle) * self.vertebra_dist

            # bilinear interpolation
            x0 = int(math.floor(x))
            y0 = int(math.floor(y))
            x1 = int(math.ceil(x))
            y1 = int(math.ceil(y))
            
            value = matrix[y0, x0]*(x1 - x)*(y1 - y) + matrix[y0, x1]*(y1 - y)*(x - x0) + matrix[y1, x0]*(x1 - x)*(y - y0) + matrix[y1, x1]*(x - x0)*(y - y0)
                    
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value                
            
            angle = angle + (math.pi / 180) / 1
            if angle > end_angle:
                break
            
         # end scan cycle
            
        return (best_x, best_y, best_value)            
        
    def align_backbone(self, matrix, weight_matrix, original_backbone, ref_value, min_vertebra, prev, min_value_coeff):

        central_value = 0
        central_index = 0
        
        max_vertebra_to_add = self.max_vertebra / 10
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
               
               (best_x, best_y, best_value) = self.align_vertebra(matrix, backbone, v, prev, prev_prev)
               
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
                           
               dx = best_x - v.center.x
               dy = best_y - v.center.y

               for v1 in backbone[idx:]:                                                  
                   v1.center.x = v1.center.x + dx
                   v1.center.y = v1.center.y + dy
                       
           # if it's the last vertebra, try to prolong the backbone...
           if idx == len(backbone) - 1 and vertebra_added < max_vertebra_to_add:             
              pvd = geometry.distance(prev.center.x, prev.center.y, v.center.x, v.center.y)
              next_center = geometry.point_along_a_line(prev.center.x, prev.center.y, v.center.x, v.center.y, pvd + self.vertebra_dist)
              backbone.append(self.Vertebra(next_center[0], next_center[1], 0))
              vertebra_added = vertebra_added + 1
            
           idx = idx + 1

        return (backbone, central_value, central_index)

    def do_track(self, matrix, weight_matrix, backbone, animals):
                                 
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
                            
        (new_front, front_val, front_index) = self.align_backbone(matrix, weight_matrix, self.backbone[cvi:], reference_value, 1, prev, self.front_min_value_coeff)

        prev = new_front[1]

        (new_back, back_val, back_index) = self.align_backbone(matrix, weight_matrix, reversed(self.backbone[:cvi + 1]), reference_value, 0, prev, self.back_min_value_coeff)

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
            
        if len(backbone) * self.vertebra_dist < 15:
            backbone = self.backbone
            
        return backbone
              
            
    def align_body_part(self, raw_matrix, weight_matrix, bp, start_x, start_y, scan_range):

        matrix1 = np.copy(raw_matrix)
        matrix1 = matrix1.astype(float)                
        matrix1 = cv2.filter2D(matrix1, -1, self.back.mass_center_filter)        
                
        scanner = self.Scanner(int(start_x), int(start_y))
           
        best_value = 0
        best_x = start_x
        best_y = start_y

        first = True
           
        while True:               
            
            if not first:                       
                (x, y) = scanner.next()
            else:
                (x, y) = (start_x, start_y)
                first = False

            dist = geometry.distance(x, y, start_x, start_y)
                   
            if dist > scan_range:
                break
            #inertia = math.pow(inertia, 1./6)
               
            value = matrix1[y, x]
                   
            if value >= best_value:
                best_x = x
                best_y = y
                best_value = value
        
        bp.center.x = best_x
        bp.center.y = best_y
            
    def track(self, raw_matrix, weight_matrix, animals):

        debug = []

        matrix = np.copy(raw_matrix)
        matrix = matrix.astype(float)        
        matrix = np.multiply(matrix, weight_matrix)
        matrix = cv2.filter2D(matrix, -1, self.mass_center_filter)        
                        
        debug_matrix = np.copy(matrix)
        cv2.normalize(debug_matrix, debug_matrix, 0, 255, cv2.NORM_MINMAX)   
        debug.append(("center of mass " + str(self.animal_number), debug_matrix))
                
        bb1 = []        
        for v in self.backbone:
            bb1.append(v.clone())
                
        self.backbone = self.do_track(matrix, weight_matrix, self.backbone, animals)

        # find countour

        blur = np.array([[1, 1, 1, 1, 1], 
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1]]) / 25.
                         
        #blurred = cv2.filter2D(raw_matrix, -1, blur)        
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
        debug.append(("body contour " + str(self.animal_number), debug_matrix))
        
            
        

        
        '''
        bbr = list(reversed(bb1))
        
        bbr = self.do_track(matrix, weight_matrix, bbr)
        
        if len(bbr) > len(self.backbone):
            self.backbone = list(reversed(bbr))
      
        # align mounts
        if self.mount != 0:
        
            matrix1 = np.copy(raw_matrix)
            matrix1 = matrix1.astype(float)                
            matrix1 = np.multiply(matrix1, weight_matrix)
        
            self.set_weights_no_mount(matrix1, 0.0)
        
            hc = self.backbone[-1].center
            
            self.head.center.x = hc.x
            self.head.center.y = hc.y
            
            hmv = self.mean_value(raw_matrix, self.head)            
            self.align_body_part(matrix1, weight_matrix, self.mount, hc.x, hc.y, self.head.radius + self.mount.radius);
            mmv = self.mean_value(matrix1, self.mount)
            self.mount_visible = mmv > 0.5 * hmv
                                                

            if self.mount_visible:
                self.mount.set_weights(matrix1, 0.0)            
                m1sx = self.mount.center.x
                m1sy = self.mount.center.y
                m1sr = self.mount.radius + self.mount1.radius
            else:
                m1sx = hc.x
                m1sy = hc.y
                m1sr = self.head.radius + self.mount1.radius
                
                
            self.align_body_part(matrix1, weight_matrix, self.mount1, m1sx, m1sy, m1sr);

            m1mv = self.mean_value(matrix1, self.mount1)
            self.mount1_visible = m1mv > 0.5 * hmv
        '''
        
        return debug
        
        '''
        cb1 = self.back.center.copy()
        cf1 = self.front.center.copy()
        ch1 = self.head.center.copy()
        cm1 = 0
        cm11 = 0

        
        if self.mount != 0:
            cm1 = self.mount.center.copy()
            
        if self.mount1 != 0:
            cm11 = self.mount1.center.copy()
        
        
        parts = [self.back, self.front, self.head]        
#        if self.mount:
#            parts.append(self.mount)


        sum1 = self.do_track(matrix, weight_matrix, parts);        
        
        '''
        '''
        rows, cols = matrix.shape[:2];
        mask = np.zeros((rows, cols), np.float);
        self.set_weights_no_mount(mask, 1.);
        mpl = np.multiply(mask, matrix);       
        sum1 = mpl.sum();               
        ''' 
        '''
                
#        sum1 = self.back.value + self.front.value + self.head.value;
#        if self.mount != 0:
#            sum1 = sum1 + self.mount.value
        
#        sum1 = self.front.value + self.head.value;
        
        cb2 = self.back.center.copy()
        cf2 = self.front.center.copy()
        ch2 = self.head.center.copy()
        cm2 = 0
        cm12 = 0
        cm2v = self.mount_visible
        cm12v = self.mount1_visible
        
        if self.mount != 0:
            cm2 = self.mount.center.copy()
        if self.mount1 != 0:
            cm12 = self.mount1.center.copy()
        
        self.back.center = cb1;
        self.front.center = cf1;
        self.head.center = ch1;
        if self.mount != 0:
            self.mount.center = cm1
        if self.mount1 != 0:
            self.mount1.center = cm11
        
        parts = [self.head, self.front, self.back]        
#        if self.mount != 0:
#            parts = [self.mount] + parts
        
        sum2 = self.do_track(matrix, weight_matrix, parts);        
        
        '''
        '''
        rows, cols = matrix.shape[:2];
        mask = np.zeros((rows, cols), np.float);
        self.set_weights_no_mount(mask, 1.);
        mpl = np.multiply(mask, matrix);       
        sum2 = mpl.sum();               
        '''
        '''
        
#        sum2 = self.back.value + self.front.value + self.head.value;
#        if self.mount != 0:
#            sum2 = sum2 + self.mount.value
        #sum2 = self.front.value + self.head.value;
        
        if sum1 > sum2:
            self.back.center = cb2
            self.front.center = cf2
            self.head.center = ch2
            if self.mount != 0:
                self.mount.center = cm2
                self.mount_visible = cm2v
            if self.mount1 != 0:
                self.mount1.center = cm12
                self.mount1_visible = cm12v
         '''  
            
class Animals:

    animals = []    
    
    def __init__(self, params):
        
        self.params = params        
    
    def add_animal(self, start_x, start_y, end_x, end_y):
        
        self.animals.append(Animal(self.params, start_x, start_y, end_x, end_y))
        return self.animals[-1]
        
    def get_positions(self):
        results = []        
        for a in self.animals:
            results.append((a, a.get_position()))
        return results            

        
    def track(self, matrix):

        rows, cols = matrix.shape[:2]
        
        debug = []
        
        weights = []

        for a in self.animals:
                                                                        
            weight = np.ones((rows, cols), np.float)                        
            
            for other in self.animals:                
                if other != a:
                    other.set_weights(weight, 0.0, 0.0)
                    
            debug_weight = np.copy(weight)
            cv2.normalize(debug_weight, debug_weight, 0, 255, cv2.NORM_MINMAX)   
            debug.append(("weights " + str(a.animal_number), debug_weight))
            
            weights.append(weight)
                    
        
        for a, w in zip(self.animals, weights):
                                                                        
            
                    
#            cv2.normalize(new_matrix, new_matrix, 0, 255, cv2.NORM_MINMAX)
                    
            
#            weights = np.ones((rows, cols), np.float)
#            weights.fill(255)
#            weights = np.multiply(weights, weight_matrix)
           
#            if idx == 0:                               
#               cv2.imshow('weights', weights);
#                idx = 1
#            else:
#                cv2.imshow('weights1', weights);
#                idx = 0

#            if a.back.mean_value == 0:                
#                a.back.mean_value = a.mean_value(matrix, a.back)                

#            mean = a.back.mean_value
            #ret, a_matrix = cv2.threshold(matrix, mean * 0.95, 255, cv2.THRESH_TOZERO)            
            #ret, a_matrix = cv2.threshold(a_matrix, mean, 255, cv2.THRESH_TRUNC)            
            #cv2.normalize(a_matrix, a_matrix, 0, 255, cv2.NORM_MINMAX)
            
 #           a_matrix = matrix
            
                                
            debug1 = a.track(matrix, w, self.animals)
            debug = debug + debug1

#            a.back.mean_value = a.mean_value(matrix, a.back)                

            '''
            parts = [a.back, a.front, a.head]
        
            if a.mount:
                parts.append(a.mount)

            rows, cols = matrix.shape[:2]

            
            (max, configuration) = a.do_fit(matrix, parts, [], weight_matrix)
        
            for part_config in configuration:
                part_config.part.center.x = part_config.x
                part_config.part.center.y = part_config.y
                part_config.part.value = part_config.value
            '''

            
#            weights.append(weight_matrix)
            #new_matrix1 = np.copy(new_matrix)
            #cv2.normalize(new_matrix, new_matrix1, 0, 255, cv2.NORM_MINMAX)
#            weights.append(new_matrix1)
#            weights.append(matrix)
       
        return debug
            
            
class TrackingFlowElement:
                
    def __init__(self, time, positions, filtered_image, debug):
        self.time = time
        self.positions = positions
        self.filtered_image = filtered_image  
        self.debug_frames = debug

def calculate_scale_factor(frame_width, frame_height):
    width = tracking_resolution_width
    height = tracking_resolution_height
    k = float(frame_width) / frame_height
    if k > float(width) / height:
        return float(width) / frame_width
    else:
        return float(height) / frame_height


class Tracking:

    tracking_params = TrackingParams()
    
    finished = False
    
    def __init__(self, video_file_name):            
        self.video = cv2.VideoCapture(video_file_name)
        frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.tracking_params.scale_factor = calculate_scale_factor(frame_width, frame_height)        
        self.animals = Animals(self.tracking_params)
        
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
        
        return self.animals.add_animal(start_x, start_y, end_x, end_y)
        
    def do_tracking(self, bg, start_frame, tracking_flow, time_to_stop, next_frame_semaphore, run_semaphore):

        pdb.set_trace()

        if not self.animals.animals:
            return
        
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)        
            
        # take current frame of the video
        ret, frame = self.video.read()

        if not ret:
            print('can\'t read the video')
            sys.exit()
                    
        # calculate the rectangle enclosing the first animal body - will use it for color band 
        # calculation

        rows, cols = frame.shape[:2]
        mask = np.zeros((rows, cols), np.uint8)

        for a in self.animals.animals:

            parts = [a.back, a.front, a.head]
            
            if a.mount:
                parts.append(a.mount)
                
            for p in parts:
                c = p.get_position()
                cv2.circle(mask, (int(c.x), int(c.y)), int(p.get_radius()), 1, -1)                    

#        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#        mask = cv2.inRange(roi, np.array((0., 0., 0.)), np.array((255., 255., 255.)))        
#        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,50.)))
        #    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,200.)))

#        roi_hist = cv2.calcHist([hsv_roi], [2], mask, [180], [0,180] )
        #roi_hist = cv2.calcHist([roi], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256] )
        hist_size = 16
        ratio = 256 / hist_size
        hist_a = cv2.calcHist([frame], [0, 1, 2], mask, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256] )                
        hist_all = cv2.calcHist([frame], [0, 1, 2], None, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256] )                
        
        hist = hist_a / (hist_all + 1)
#        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
#        cv2.normalize(hist, hist, 0., 1., cv2.NORM_MINMAX)
    
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        # term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        sharpen = np.array([[-1, -1, -1, -1, -1], 
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 8, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]]) / 8.0
                            
#        mask = cv2.inRange(bg, np.array((0., 0., 0.)), np.array((55., 55., 55.)))                                            
        mask = cv2.inRange(bg, np.array((0., 0., 0.)), np.array((75., 75., 75.)))
                    
            
        while(not time_to_stop.isSet()):
               
           if not run_semaphore.isSet():
               next_frame_semaphore.wait()
               next_frame_semaphore.clear()               

#           frame = cv2.calcBackProject([frame], [0, 1, 2], roi_hist, [0, 256, 0, 256, 0, 256], 1)  
#           frame = cv2.calcBackProject([frame], [0, 1, 2], hist, [0, 256, 0, 256, 0, 256], 255. / hist.max())   
#           frame1 = cv2.calcBackProject([frame], [0, 1, 2], hist, [0, 256, 0, 256, 0, 256], 1.0)  
           
           '''
           c1, c2, c3 = cv2.split(frame)
           frame1 = hist[c1.ravel() / ratio, c2.ravel() / ratio, c3.ravel() / ratio]
           frame1 = np.minimum(frame1, 1)
           frame1 = frame1.reshape(frame.shape[:2])               
           
           disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
           cv2.filter2D(frame1,-1, disc,frame1)
           frame1 = np.uint8(frame1)
           cv2.normalize(frame1, frame1, 0, 255, cv2.NORM_MINMAX)
           ret, frame1 = cv2.threshold(frame1, 1, 255, cv2.THRESH_BINARY)
           
           #cv2.normalize(frame1, frame1, 0, 255, cv2.NORM_MINMAX)
           #ret, frame1 = cv2.threshold(frame1, 50, 255, cv2.THRESH_BINARY)
                          
           frame = resize(frame1)
           
           border = tracking_border   


           frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, 0)
           '''

           debug = []
           
           frame = cv2.absdiff(frame, bg)
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#           frame = 255 - frame

#           sharpen = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
#           sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                               
#           frame = cv2.filter2D(frame, -1, sharpen)
           
           cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
#           ret, frame = cv2.threshold(frame, 20, 255, cv2.THRESH_TOZERO)
#           ret, frame = cv2.threshold(frame, 60, 255, cv2.THRESH_TOZERO)
#           ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
           
#           frame = frame + mask/5
           
           frame1 = resize(frame)
           
           border = tracking_border   

#           frame = resize(frame)

           frame = cv2.copyMakeBorder(frame1, border, border, border, border, cv2.BORDER_CONSTANT, 0)
           
           
           
           
#           frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, (0, 0, 0))
  
#           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           
#           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#           frame = (255 - frame)
                     
           '''
           cv::GaussianBlur(frame, image, cv::Size(0, 0), 3);
           cv::addWeighted(frame, 1.5, image, -0.5, 0, image);
           '''
           
#           frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0)

           
           
           #ret, frame = cv2.threshold(frame, 190, 255, cv2.THRESH_TOZERO)
           
#           cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
           
#           ret, frame = cv2.threshold(frame, 215, 255, cv2.THRESH_BINARY)

           
           # frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           
#           dst = cv2.calcBackProject([frame], [0], roi_hist, [0,255], 1)  
           #mask = cv2.inRange(hsv_roi, np.array((0.)), np.array((255.)))        
           
#           dst = frame
    
           rows, cols = frame.shape[:2]
           
           debug.append(("source", frame))
    
           debug1 = self.animals.track(frame)
           
           debug = debug + debug1

           frame_time = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.       
           
           tracking_flow_element = TrackingFlowElement(frame_time, self.animals.get_positions(), frame1, debug)       
                      
           # read the next frame
           ret, frame = self.video.read()    
           if ret == False:
               self.finished = True
                                 
           tracking_flow.put(tracking_flow_element)
 
           
           if ret == False:
               break
       
             

#        cv2.destroyAllWindows()
  
    
    