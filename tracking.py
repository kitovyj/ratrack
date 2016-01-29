import sys
import numpy as np
import cv2
import math
import threading
import Queue
import pdb

tracking_border = 20
tracking_resolution_width = 320
tracking_resolution_height = 240

curr_cos = 0

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
    
    class Center:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
    def __init__(self, params, x, y, radius):
        self.params = params
        self.center = self.Center(x * params.scale_factor + tracking_border, 
                                  y * params.scale_factor + tracking_border)
        self.original_radius = radius
        self.radius = radius * params.scale_factor

    def set_weights(self, matrix):        
        #print(self.center.x)
        #print(self.center.y)
        #print(self.radius)
        cv2.circle(matrix, (int(self.center.x), int(self.center.y)), int(self.radius), (0.0), -1)
        
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
            self.center.x = sum_x / values_sum;
            self.center.y = sum_y / values_sum;
            
    def belongs_to(self, given_center_x, given_center_y, x, y):
        distance_from_center = math.sqrt((given_center_x - x)**2 + (given_center_y - y)**2)
        return distance_from_center < self.radius
        
    # todo: add mitex synchronization to preperties retrieval methods
        
    def get_center(self):
        return self.Center((self.center.x - tracking_border) / self.params.scale_factor, 
                           (self.center.y - tracking_border) / self.params.scale_factor)

    def get_radius(self):
        return self.original_radius
        

def point_along_a_line(start_x, start_y, end_x, end_y, distance):

    dx = start_x - end_x
    dy = start_y - end_y
    x = 0
    y = 0
    if dx != 0:
        k = float(dy) / float(dx)
        point_dx = math.sqrt(distance**2 / (1 + k**2))
        if dx < 0:
            point_dx *= -1
        point_dy = point_dx * k        
        x = end_x + point_dx
        y = end_y + point_dy        
    else:
        x = end_x
        if dy < 0:
            y = end_y - distance
        else:
            y = end_y + distance
    return (x, y)                 

animal_n = 0

class Animal:
    
    class PartConfiguration:
        def __init__(self, x, y, part, val):
            self.part = part
            self.x = x;
            self.y = y;
            self.value = val; # cover value

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

        back_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(back_radius) / total)
        front_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(2*back_radius + front_radius) / total)
        head_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(2*back_radius + 2*front_radius + head_radius) / total)

        mount_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(2*back_radius + 2*front_radius + 2*head_radius + mount_radius) / total)


        if animal_n == 0:
            self.mount = BodyPart(params, int(mount_position[0]), int(mount_position[1]), mount_radius)
        else:
            self.mount = 0

        self.head = BodyPart(params, int(head_position[0]), int(head_position[1]), head_radius)
        self.front = BodyPart(params, int(front_position[0]), int(front_position[1]), front_radius)
        self.back = BodyPart(params, int(back_position[0]), int(back_position[1]), back_radius)

        animal_n = animal_n + 1;
    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)        

    def set_weights(self, matrix):        
        self.front.set_weights(matrix)        
        self.back.set_weights(matrix)
        self.head.set_weights(matrix)

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
        inner = self.weight(matrix, center_x, center_y, part.radius, weight_matrix)
        inner1 = self.weight(matrix, center_x, center_y, part.radius, minus_weight_matrix)
        outer = self.weight(matrix, center_x, center_y, part.radius + 4, minus_weight_matrix)        
        r =  inner
        r = inner - 0.5*(outer - inner1)        
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

    def cosine(self, x1, y1, x2, y2, x3, y3)                    :
        sx1 = x1 - x2
        sy1 = y1 - y2
        sx2 = x3 - x2
        sy2 = y3 - y2
        dot = sx1 * sx2 + sy1 * sy2
        a = math.sqrt(sx1**2 + sy1**2)
        b = math.sqrt(sx2**2 + sy2**2)
        if a * b != 0:
            cos = dot / (a * b)
            return cos
        else:
            return 0        

    def do_track(self, matrix, weight_matrix, parts):

        #print(self.cosine(self.back.center.x, self.back.center.y, self.front.center.x, self.front.center.y, self.head.center.x, self.head.center.y))

        new_weight_matrix = np.copy(weight_matrix)

#        print('aaaa\n')
#        pdb.set_trace()
#        max_diff = 0.001
        max_diff = 0.0001

        for idx, p in enumerate(parts):    
    
           p.dx = 0
           p.dy = 0

           minus_weight_matrix = np.copy(weight_matrix)           
           for p1 in parts:    
               if p1 != p:
                   cv2.circle(minus_weight_matrix, (p1.center.x, p1.center.y), int(p1.radius), 0.0, -1)

#           new_value = self.cover(matrix, int(p.center.x + p.vx), int(p.center.y + p.vy), p, new_weight_matrix)
           new_value = self.cover(matrix, int(p.center.x), int(p.center.y), p, new_weight_matrix, minus_weight_matrix)
           delta = new_value - p.value           
           
           min_dist = 3
           max_dist = 6
           
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
               while True:
                   
                   (x, y) = scanner.next()
                   
                   dist = math.sqrt((p.center.x - x)**2 + (p.center.y - y)**2)
                   
                   delta = 0

                   if dist <= max_dist:
                       
                       if idx > 0:
                           prev = parts[idx - 1]
                           dst = math.sqrt((prev.center.x - x)**2 + (prev.center.y - y)**2)
                           if dst - (prev.radius + p.radius) > 1.5:
                               continue
                                          
                       if idx == 2:
                           cos = self.cosine(parts[idx - 2].center.x, parts[idx - 2].center.y, parts[idx - 1].center.x, parts[idx - 1].center.y, x, y)
                           if cos > 0.1:
                               continue
#                           else:
#                               print(cos)
                       if idx == 1:
                           dx = float(x - p.center.x)
                           dy = float(y - p.center.y)                           
                           cos = self.cosine(parts[idx - 1].center.x, parts[idx - 1].center.y, x, y, parts[idx + 1].center.x + dx, parts[idx + 1].center.y + dy)
                           if cos > 0.1:
                               continue                               

                       new_vx = float(x - p.center.x)
                       new_vy = float(y - p.center.y)
                   
                   #pl = 0.02;
            #       pl = 0.01;
#                   pl = 0.01;
                       pl = 0.01
#                   pl = 0.0005
                       pl = 0.003
                   
                      # delta_v = math.sqrt((p.vx - new_vx)**2 + (p.vy - new_vy)**2)
                       delta_v = math.sqrt((new_vx)**2 + (new_vy)**2)
                       pk = (1 - delta_v*pl)**3

                       minus_weight_matrix = np.copy(weight_matrix)           
                       for p1 in parts:
                           if p1 != p:
                               cv2.circle(minus_weight_matrix, (int(round(p1.center.x + new_vx)), int(round(p1.center.y + new_vy))), int(p1.radius), 0.0, -1)
                   
                       new_value = self.cover(matrix, x, y, p, new_weight_matrix, minus_weight_matrix) * pk

                       if new_value > best_val:
                           best_val = new_value
                           best_x = x
                           best_y = y
                       
                       
                       delta = new_value - p.value           
                   
#                   if dist > min_dist and (dist > max_dist or delta > 0 or (- delta / p.value) < max_diff):
#                   if dist > min_dist and (dist > max_dist or delta > 0 or (- delta / p.value) < max_diff):
                   if dist > max_dist:
                   
                       # or (dist > 2. and (- delta / p.value) < 0.05 * dist):
                       
                       if dist > max_dist:
                           new_value = best_val
                           x = best_x
                           y = best_y
                       
                       new_vx = float(x - p.center.x)
                       new_vy = float(y - p.center.y)
                       delta_v = math.sqrt((p.vx - new_vx)**2 + (p.vy - new_vy)**2)
                       pk = (1 - delta_v*pl)
                       p.value = new_value / pk
                       
#                       if p == self.back:
#                           print(new_value)
#                           sys.stdout.flush()
                           
                       p.dx = new_vx
                       p.dy = new_vy
#                       p.vx = new_vx
#                       p.vy = new_vy                           
                       break                
           else:
               #if delta > 0:
               p.value = new_value
                   
           for p1 in parts[idx:]:                                                  
               p1.center.x = int(round(p1.center.x + p.dx))
               p1.center.y = int(round(p1.center.y + p.dy))
               
           cv2.circle(new_weight_matrix, (p.center.x, p.center.y), int(p.radius), (0.9), -1)                    
            
            
    def track(self, matrix, weight_matrix):

        cb1 = self.back.center
        cf1 = self.front.center
        ch1 = self.head.center
        cm1 = 0
        
        if self.mount != 0:
            cm1 = self.mount.center
        
        
        parts = [self.back, self.front, self.head]        
        if self.mount:
            parts.append(self.mount)


        self.do_track(matrix, weight_matrix, parts);        
        
        sum1 = self.back.value + self.front.value + self.head.value;
        if self.mount != 0:
            sum1 = sum1 + self.mount.value
        
#        sum1 = self.front.value + self.head.value;
        
        cb2 = self.back.center
        cf2 = self.front.center
        ch2 = self.head.center
        cm2 = 0
        
        if self.mount != 0:
            cm2 = self.mount.center
        
        self.back.center = cb1;
        self.front.center = cf1;
        self.head.center = ch1;
        if self.mount != 0:
            self.mount.center = cm1
        
        parts = [self.head, self.front, self.back]        
        if self.mount != 0:
            parts = [self.mount] + parts
        
        self.do_track(matrix, weight_matrix, parts);        
        sum2 = self.back.value + self.front.value + self.head.value;
        if self.mount != 0:
            sum2 = sum2 + self.mount.value
        #sum2 = self.front.value + self.head.value;
        
        if sum1 > sum2:
            self.back.center = cb2;
            self.front.center = cf2;
            self.head.center = ch2;
            if self.mount != 0:
                self.mount.center = cm2
                
            
class Animals:

    animals = []    
    
    def __init__(self, params):
        
        self.params = params        
    
    def add_animal(self, start_x, start_y, end_x, end_y):
        
        self.animals.append(Animal(self.params, start_x, start_y, end_x, end_y))
        return self.animals[-1]
        
    def track(self, matrix):

        rows, cols = matrix.shape[:2]
        
        weights = []
        
        for a in self.animals:
            
            weight_matrix = np.ones((rows, cols), np.float)
            
            for other in self.animals:                
                if other != a:
                    other.set_weights(weight_matrix)
                    
            
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
            
            weights.append(weight_matrix)
        
        return weights
            
            
class TrackingFlowElement:
                
    def __init__(self, filtered_image, weights):
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
        
        video = cv2.VideoCapture(video_file)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)        
            
        # take current frame of the video
        ret, frame = video.read()

        if not ret:
            print('can\'t read the video')
            sys.exit()
                    
        # calculate the rectangle enclosing the first animal body - will use it for color band 
        # calculation

        animal = self.animals.animals[0]

        left_x, right_x, top_y, bottom_y = 0, 0, 0, 0

        back_center = animal.back.get_center()
        back_radius = animal.back.get_radius()
        head_center = animal.head.get_center()
        head_radius = animal.head.get_radius()
                
        if head_center.x < back_center.x:
            left_x = head_center.x - head_radius
            right_x = back_center.x + back_radius
        else:
            right_x = head_center.x + head_radius
            left_x = back_center.x - back_radius

        if head_center.y < back_center.y:
            top_y = head_center.y - head_radius
            bottom_y = back_center.y + back_radius
        else:
            bottom_y = head_center.y + head_radius
            top_y = back_center.y - back_radius
    
        roi = frame[top_y:bottom_y, left_x:right_x]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_roi, np.array((0.)), np.array((255.)))        
#        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,50.)))
        #    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,200.)))

#        roi_hist = cv2.calcHist([hsv_roi], [2], mask, [180], [0,180] )
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [255], [0,255] )
        
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            
        while(not time_to_stop.isSet()):
               
           if not run_semaphore.isSet():
               next_frame_semaphore.wait()
               next_frame_semaphore.clear()               
               
           frame = resize(frame)

           border = tracking_border   
           frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, (0, 0, 0))
  
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

           frame = (255 - frame)           
#           frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0)
#           ret, frame = cv2.threshold(frame, 215, 255, cv2.THRESH_BINARY)
#           ret, frame = cv2.threshold(frame, 215, 255, cv2.THRESH_BINARY)

           
           # frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           
#           dst = cv2.calcBackProject([frame], [2], roi_hist, [0,180], 1)  
#           dst = cv2.calcBackProject([frame], [0], roi_hist, [0,255], 1)  
           #mask = cv2.inRange(hsv_roi, np.array((0.)), np.array((255.)))        
           
#           dst = frame
    
           rows, cols = frame.shape[:2]
    
           weights = self.animals.track(frame)
                                       
           tracking_flow_element = TrackingFlowElement(frame, weights)       
           tracking_flow.join()
           tracking_flow.put(tracking_flow_element)

           # read the next frame
           ret, frame = video.read()    

           if ret == False:
               break
       
             

        cv2.destroyAllWindows()
  
    
    
