import sys
import numpy as np
import cv2
import math
import threading
import Queue
import pdb
import geometry

from geometry import Point

tracking_border = 20
tracking_resolution_width = 320
tracking_resolution_height = 240

curr_cos = 0

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
        
    def __init__(self, params, x, y, radius):
        self.params = params
        self.center = Point(x * params.scale_factor + tracking_border, 
                            y * params.scale_factor + tracking_border)
        self.original_radius = radius
        self.radius = radius * params.scale_factor

    def set_weights(self, matrix, weight):        
        #print(self.center.x)
        #print(self.center.y)
        #print(self.radius)
        cv2.circle(matrix, (int(self.center.x), int(self.center.y)), int(self.radius), weight, -1)
        
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
                
animal_n = 0

class AnimalPosition:
    def __init__(self):
        self.head = 0
        self.front = 0
        self.back = 0
        self.mount = 0

class Animal:
    
    class PartConfiguration:
        def __init__(self, x, y, part, val):
            self.part = part
            self.x = x
            self.y = y
            self.value = val # cover value

    # deduce body parts positions from back-to-front vector
    def __init__(self, params, start_x, start_y, end_x, end_y):
        
        self.params = params        
        
        global animal_n

        head_radius = 6 / params.scale_factor
        front_radius = 7 / params.scale_factor
        back_radius = 9 / params.scale_factor
        mount_radius = 6 / params.scale_factor

        
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
            total = total + 2*mount_radius

        back_position = geometry.point_along_a_line(end_x, end_y, start_x, 
                                                    start_y, length * float(back_radius) / total)
        front_position = geometry.point_along_a_line(end_x, end_y, 
                                                     start_x, start_y, length * float(2*back_radius + front_radius) / total)
        head_position = geometry.point_along_a_line(end_x, end_y, 
                                                    start_x, start_y, length * float(2*back_radius + 2*front_radius + head_radius) / total)

        mount_position = geometry.point_along_a_line(end_x, end_y, 
                                                     start_x, start_y, length * float(2*back_radius + 2*front_radius + 2*head_radius + mount_radius) / total)

        if animal_n == 0:
            self.mount = BodyPart(params, int(mount_position[0]), int(mount_position[1]), mount_radius)
        else:
            self.mount = 0

        self.head = BodyPart(params, int(head_position[0]), int(head_position[1]), head_radius)
        self.front = BodyPart(params, int(front_position[0]), int(front_position[1]), front_radius)
        self.back = BodyPart(params, int(back_position[0]), int(back_position[1]), back_radius)

        animal_n = animal_n + 1;
        
    def get_position(self):        
        r = AnimalPosition()
        r.head = self.head.get_position()
        r.front = self.front.get_position()
        r.back = self.back.get_position()
        if self.mount:
            r.mount = self.mount.get_position()
        return r
                                    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)        

    def set_weights_no_mount(self, matrix, weight):        
        self.front.set_weights(matrix, weight)        
        self.back.set_weights(matrix, weight)
        self.head.set_weights(matrix, weight)
        
        hc = self.head.center
        fc = self.front.center
        bc = self.back.center
        hr = self.head.radius
        fr = self.front.radius
        br = self.back.radius
        
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
        

    def set_weights(self, matrix, weight, mount_weight):        
        if self.mount != 0:
            self.mount.set_weights(matrix, weight)
        self.set_weights_no_mount(matrix, weight)

    def weight(self, matrix, center_x, center_y, radius, weight_matrix):

        mask = np.zeros((int(radius*2) + 1, int(radius*2) + 1), np.float)        
        cv2.circle(mask, (int(radius), int(radius)), int(radius), (1.0), -1)                    
        masked = np.multiply(mask, matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])
        weighted =  np.multiply(masked, weight_matrix[int(center_y - radius): int(center_y + radius) + 1, int(center_x - radius):int(center_x + radius) + 1])
        
        return weighted.sum();


    def bfcover(self, matrix, center_x, center_y, part, weight_matrix):        
        inner = self.weight(matrix, center_x, center_y, part.radius, weight_matrix)
        return inner

    def cover(self, matrix, center_x, center_y, part, weight_matrix, minus_weight_matrix):
        outer_r = part.radius + 2
        area_inner = math.pi * (part.radius**2)
        area_outer = math.pi * (outer_r**2)
        inner = self.weight(matrix, center_x, center_y, part.radius, weight_matrix)
#        inner1 = self.weight(matrix, center_x, center_y, part.radius, minus_weight_matrix)
#        outer = self.weight(matrix, center_x, center_y, outer_r, minus_weight_matrix)        
        r =  inner

#        r =  inner - 0.5 * (outer - inner1)
                
                
#        r = (inner / area_inner)
#        r = r / (300. + ((outer - inner1) / (area_outer - area_inner)) )  
        
        return r;        
        

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

    def do_track(self, matrix, weight_matrix, parts):

        #print(geometry.cosine(self.back.center.x, self.back.center.y, self.front.center.x, self.front.center.y, self.head.center.x, self.head.center.y))

        new_weight_matrix = np.copy(weight_matrix)

#        print('aaaa\n')
#        pdb.set_trace()
#        max_diff = 0.001
        max_diff = 0.0001

        for idx, p in enumerate(parts):    
    
           p.dx = 0
           p.dy = 0
           
           minus_weight_matrix = np.copy(weight_matrix)           

#           for p1 in parts:    
#               if p1 != p:
#                   cv2.circle(minus_weight_matrix, (p1.center.x, p1.center.y), int(p1.radius), 0.0, -1)

#           if self.mount != 0:
#               cv2.circle(minus_weight_matrix, (int(round(self.mount.center.x)), int(round(self.mount.center.y))), int(self.mount.radius), 0.0, -1)
           
#           new_value = self.cover(matrix, int(p.center.x + p.vx), int(p.center.y + p.vy), p, new_weight_matrix)
           
           if p == self.front:                           
             w = 0.5
             cv2.circle(new_weight_matrix, (self.back.center.x, self.back.center.y), int(self.back.radius), w, -1)
           elif p == self.head:
             w = 0.5
             cv2.circle(new_weight_matrix, (self.back.center.x, self.back.center.y), int(self.back.radius), w, -1)
             cv2.circle(new_weight_matrix, (self.front.center.x, self.front.center.y), int(self.front.radius), w, -1)
           
           new_value = self.cover(matrix, int(p.center.x), int(p.center.y), p, new_weight_matrix, minus_weight_matrix)
           delta = new_value - p.value           
           p.value = new_value
                      
           min_dist = 2
           max_dist = 5

           if p == self.head:                           
               max_dist = 7
           elif p == self.front:
               max_dist = 7
           else:
               max_dist = 7
             
           
           if idx > 0:
               prev = parts[idx - 1]
               dfp = math.sqrt((p.center.x - prev.center.x)**2 + (p.center.y - prev.center.y)**2)
               if dfp < (prev.radius + p.radius):
                   min_dist = ((prev.radius + p.radius) - dfp)
                   
           if min_dist >= max_dist:
               max_dist = min_dist + 1
                      
           if min_dist > 0 or (delta <  0. and (- delta / p.value) > max_diff) :
               
              # print('bbb')
               
#               scanner = self.Scanner(int(p.center.x + p.vx), int(p.center.y + p.vy))
               scanner = self.Scanner(int(p.center.x), int(p.center.y))
               best_x = p.center.x
               best_y = p.center.y
               
               best_val = new_value
               best_pk = 1.
               
               while True:
                   
                   (x, y) = scanner.next()

                   dist = math.sqrt((p.center.x - x)**2 + (p.center.y - y)**2)

                   '''
                   if dist <= max_dist:                                              

                       #if p == self.front:
                       #    was = geometry.distance(self.head.center.x, self.head.center.y, self.front.center.x, self.front.center.y)
                       #    now = geometry.distance(self.head.center.x, self.head.center.y, x, y)
                       #    if now > was:
                       #        continue
                                 
                       
                       if p == self.back:
                           was = geometry.distance(self.front.center.x, self.front.center.y, self.back.center.x, self.back.center.y)
                           now = geometry.distance(self.front.center.x, self.front.center.y, x, y)
                           if now > was:
                               continue
                   '''
                                                                               
                   
                   delta = 0

                   if dist <= max_dist:
                       
                       if idx > 0:
                           prev = parts[idx - 1]
                           dst = math.sqrt((prev.center.x - x)**2 + (prev.center.y - y)**2)
                           if dst - (prev.radius + p.radius) > 1.5:
                               continue
                       
                       if idx == 2:
                           cos = geometry.cosine(parts[idx - 2].center.x, parts[idx - 2].center.y, parts[idx - 1].center.x, parts[idx - 1].center.y, x, y)
                           if cos > 0.1:
                               continue
#                           else:
#                               print(cos)
                       if idx == 1:
                           dx = float(x - p.center.x)
                           dy = float(y - p.center.y)                           
                           cos = geometry.cosine(parts[idx - 1].center.x, parts[idx - 1].center.y, x, y, parts[idx + 1].center.x + dx, parts[idx + 1].center.y + dy)
                           if cos > 0.1:
                               continue                               
                                                                     
                       rotation_coeff = 1.
 
                       if idx > 0:
                           
                           rotation_probability_front = [ 1.0, 0.9, 0.9, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 
                                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0 ]
                                 
                           rotation_probability_head = [ 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.3, 
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0]
                                                         
                           rotation_probability = rotation_probability_front
                           
                           if p == self.head:
                               rotation_probability = rotation_probability_head                                                        
                                 
                           prev = parts[idx - 1]                   
                           stretch = geometry.distance(x, y, prev.center.x, prev.center.y) 
                           if stretch == 0: # can't be
                               continue
                           cosine = geometry.cosine(p.center.x, p.center.y, prev.center.x, prev.center.y, x, y)          
                           #print cosine
                           rotation_coeff = rotation_probability[int(min(round(math.acos(cosine) / (math.pi / 18)), 17))]
#                           if prev == self.back and p == self.front:
#                               print math.acos(cosine) * 180 / math.pi
                           #print rotation_coeff
                           #if cosine < diff * 0.7 * math.sqrt((prev.radius + p.radius) / stretch):
#                           if cosine < diff * 0.7 * math.pow((prev.radius + p.radius) / stretch, 1 / 4.):
#                               continue

                       stretch_coeff = 1.                    
                       motion_coeff = 1.

                       if idx > 0:
                           stretch_probability = [ 1.0, 0.97, 0.9, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0,0 ]                           
                           prev = parts[idx - 1]                   
                           was = geometry.distance(p.center.x, p.center.y, prev.center.x, prev.center.y) 
                           now = geometry.distance(x, y, prev.center.x, prev.center.y) 
                           stretch = abs(was - now)
                           stretch_coeff = stretch_probability[int(min(stretch, 9))]
                       else:
                           motion_probability = [ 1.0, 0.97, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0,0 ]                           
                           motion = geometry.distance(p.center.x, p.center.y, x, y) 
                           motion_coeff = motion_probability[int(min(motion, 9))]

                       pk = rotation_coeff * stretch_coeff * motion_coeff
                       
                       if pk < 0.1:
                           continue
                           
                       new_vx = float(x - p.center.x)
                       new_vy = float(y - p.center.y)

                       minus_weight_matrix = np.copy(weight_matrix)           
#                       for p1 in parts:
#                           if p1 != p:
#                               cv2.circle(minus_weight_matrix, (int(round(p1.center.x + new_vx)), int(round(p1.center.y + new_vy))), int(p1.radius), 0.0, -1)
                       
 #                      if self.mount != 0:
#                           cv2.circle(minus_weight_matrix, (int(round(self.mount.center.x)), int(round(self.mount.center.y))), int(self.mount.radius), 0.0, -1)

                       cover = self.cover(matrix, x, y, p, new_weight_matrix, minus_weight_matrix)
                                            
                       new_value = cover * pk
                       
                       if new_value > best_val:
                           best_val = new_value
                           best_x = x
                           best_y = y
                           best_pk = pk
                                                  
                       
                       delta = new_value - p.value           
                   
#                   if dist > min_dist and (dist > max_dist or delta > 0 or (- delta / p.value) < max_diff):
#                   if dist > min_dist and (dist > max_dist or delta > 0 or (- delta / p.value) < max_diff):
                   if dist > max_dist:
                   
                       # or (dist > 2. and (- delta / p.value) < 0.05 * dist):
                       
                       if dist > max_dist:
                           new_value = best_val
                           x = best_x
                           y = best_y
                       
                       
                       p.value = new_value / best_pk
                       
#                       if p == self.back:
#                           print(new_value)
#                           sys.stdout.flush()
                           
                       p.dx = x - p.center.x
                       p.dy = y - p.center.y
#                       p.vx = new_vx
#                       p.vy = new_vy                           
                       break                
           else:
               #if delta > 0:
               p.value = new_value
                   
           for p1 in parts[idx:]:                                                  
               p1.center.x = int(round(p1.center.x + p.dx))
               p1.center.y = int(round(p1.center.y + p.dy))
               
               
                   
        if self.mount == 0:
            return
        
        '''
        minus_weight_matrix = np.copy(weight_matrix)           
        self.set_weights_no_mount(minus_weight_matrix, 0.0)
        '''
        
        new_weight_matrix = np.copy(weight_matrix)           
        self.set_weights_no_mount(new_weight_matrix, 0.0)
                   
 #       start_x = self.mount.center.x + self.head.dx
#        start_y = self.mount.center.y + self.head.dy
        start_x = self.head.center.x
        start_y = self.head.center.y
        
        scanner = self.Scanner(int(start_x), int(start_y))
        new_value = self.cover(matrix, int(start_x), int(start_y), self.mount, new_weight_matrix, new_weight_matrix)

        best_x = start_x
        best_y = start_y
        best_val = new_value
        
        max_dist = self.head.radius*2
#        max_dist = 4
        
        while True:                   

            (x, y) = scanner.next()

            dist_from_head = math.sqrt((self.head.center.x - x)**2 + (self.head.center.y - y)**2)
            if dist_from_head > self.head.radius + self.mount.radius + 2:
                break
#                continue
            
            new_value = self.cover(matrix, x, y, self.mount, new_weight_matrix, new_weight_matrix)

            if new_value > best_val:
                best_val = new_value
                best_x = x
                best_y = y
                                               
            dist = math.sqrt((start_x - x)**2 + (start_y - y)**2)
            
            if dist > max_dist:
                break            
                        
            
        self.mount.center.x = best_x
        self.mount.center.y = best_y
        
        
            
    def track(self, matrix, weight_matrix):

        cb1 = self.back.center.copy()
        cf1 = self.front.center.copy()
        ch1 = self.head.center.copy()
        cm1 = 0
        
        if self.mount != 0:
            cm1 = self.mount.center.copy()
        
        
        parts = [self.back, self.front, self.head]        
#        if self.mount:
#            parts.append(self.mount)


        self.do_track(matrix, weight_matrix, parts);        
        
        
        rows, cols = matrix.shape[:2];
        mask = np.zeros((rows, cols), np.float);
        self.set_weights_no_mount(mask, 1.);
        mpl = np.multiply(mask, matrix);       
        sum1 = mpl.sum();               
        
                
#        sum1 = self.back.value + self.front.value + self.head.value;
#        if self.mount != 0:
#            sum1 = sum1 + self.mount.value
        
#        sum1 = self.front.value + self.head.value;
        
        cb2 = self.back.center.copy()
        cf2 = self.front.center.copy()
        ch2 = self.head.center.copy()
        cm2 = 0
        
        if self.mount != 0:
            cm2 = self.mount.center.copy()
        
        self.back.center = cb1;
        self.front.center = cf1;
        self.head.center = ch1;
        if self.mount != 0:
            self.mount.center = cm1
        
        parts = [self.head, self.front, self.back]        
#        if self.mount != 0:
#            parts = [self.mount] + parts
        
        self.do_track(matrix, weight_matrix, parts);        
        
        rows, cols = matrix.shape[:2];
        mask = np.zeros((rows, cols), np.float);
        self.set_weights_no_mount(mask, 1.);
        mpl = np.multiply(mask, matrix);       
        sum2 = mpl.sum();               
        
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
        
        weights = []
        
        for a in self.animals:
            
            weight_matrix = np.ones((rows, cols), np.float)
            
            for other in self.animals:                
                if other != a:
                    other.set_weights(weight_matrix, 0.0, 0.0)
                    
            
#            weights = np.ones((rows, cols), np.float)
#            weights.fill(255)
#            weights = np.multiply(weights, weight_matrix)
           
#            if idx == 0:                               
#               cv2.imshow('weights', weights);
#                idx = 1
#            else:
#                cv2.imshow('weights1', weights);
#                idx = 0
                
                    
            a.track(matrix, weight_matrix)

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

            
            weights.append(weight_matrix)
        
        return weights
            
            
class TrackingFlowElement:
                
    def __init__(self, positions, filtered_image, weights):
        self.positions = positions
        self.filtered_image = filtered_image  
        self.weights = weights          

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
    
    def __init__(self, frame_width, frame_height):            
        self.tracking_params.scale_factor = calculate_scale_factor(frame_width, frame_height)        
        self.animals = Animals(self.tracking_params)

    def add_animal(self, start_x, start_y, end_x, end_y):
        
        return self.animals.add_animal(start_x, start_y, end_x, end_y)
        
    def do_tracking(self, video_file, start_frame, tracking_flow, time_to_stop, next_frame_semaphore, run_semaphore):

        pdb.set_trace()

        if not self.animals.animals:
            return

        '''
        video = cv2.VideoCapture(video_file)
        ret, frame = video.read()
        rows, cols = frame.shape[:2]
        bg = np.zeros((rows, cols, 3), np.uint32)
        total = 0
        while True:        
            bg = bg + frame            
            total = total + 1
            ret, frame = video.read()
            if ret == False:
                break
            #if total > 10000:
#                break
        
        bg = bg / total

        bg = bg.astype(np.uint8)
        
        cv2.imwrite('bg.tiff', bg)
        '''

        bg = cv2.imread('bg.tiff')
        
        video = cv2.VideoCapture(video_file)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)        
            
        # take current frame of the video
        ret, frame = video.read()

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
           
           frame = cv2.absdiff(frame, bg)
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#           frame = 255 - frame

#           sharpen = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
#           sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                               
#           frame = cv2.filter2D(frame, -1, sharpen)
           
           ret, frame = cv2.threshold(frame, 60, 255, cv2.THRESH_TOZERO)
#           cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
#           ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
           
#           frame = frame + mask/5
           
           frame = resize(frame)
           
           border = tracking_border   

#           frame = resize(frame)

           frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, 0)
           
           
           
           
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
    
           weights = self.animals.track(frame)
                                       
           tracking_flow_element = TrackingFlowElement(self.animals.get_positions(), frame, weights)       
           tracking_flow.put(tracking_flow_element)
 
           # read the next frame
           ret, frame = video.read()    

           if ret == False:
               break
       
             

#        cv2.destroyAllWindows()
  
    
    
