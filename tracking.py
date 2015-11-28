import sys
import numpy as np
import cv2
import math
import threading
import Queue

tracking_border = 20
tracking_resolution_width = 320
tracking_resolution_height = 240

class TrackingParams:
    
    scale_factor = 1    
    
    def __init__(self, scale_factor = 1):
        self.scale_factor = scale_factor

class BodyPart:
        
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
        cv2.circle(matrix, (self.center.x, self.center.y), self.radius, (0.5), -1)
        
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
        return self.Center(self.center.x / self.params.scale_factor - tracking_border, 
                           self.center.y / self.params.scale_factor - tracking_border)

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

class Animal:
    
    class PartConfiguration:
        def __init__(self, x, y, part):
            self.part = part
            self.x = x;
            self.y = y;        

    # deduce body parts positions from back-to-front vector
    def __init__(self, params, start_x, start_y, end_x, end_y):
        
        self.params = params        
        
        head_radius = 5 / params.scale_factor
        front_radius = 6 / params.scale_factor
        back_radius = 8 / params.scale_factor

#        head_radius = 3
#        front_radius = 5
#        back_radius = 7
                
        length = math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
        
        total = 2*back_radius + 2*front_radius + 2*head_radius

        back_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(back_radius) / total)
        front_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(2*back_radius + front_radius) / total)
        head_position = point_along_a_line(start_x, start_y, end_x, end_y, length * float(2*back_radius + 2*front_radius + head_radius) / total)

        self.head = BodyPart(params, int(head_position[0]), int(head_position[1]), head_radius)
        self.front = BodyPart(params, int(front_position[0]), int(front_position[1]), front_radius)
        self.back = BodyPart(params, int(back_position[0]), int(back_position[1]), back_radius)        
    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)
            
        """
        # deduce head position
        dx = self.front.center.x - self.back.center.x
        dy = self.front.center.y - self.back.center.y
        head_x = 0
        head_y = 0
        head_radius = 3
        if dx != 0:
            k = float(dy) / float(dx)
            head_dx = math.sqrt((head_radius + self.front.radius)**2 / (1 + k**2))
            if dx < 0:
                head_dx *= -1
            head_dy = head_dx * k        
            head_x = self.front.center.x + head_dx
            head_y = self.front.center.y + head_dy        
        else:
            head_x = self.front.center.x
            if dy < 0:
                head_y = self.front.center.y - (head_radius + self.front.radius)
            else:
                head_y = self.front.center.y + (head_radius + self.front.radius)
            
        cv2.circle(matrix, (int(head_x), int(head_y)), head_radius, (0, 0, 255))
        """    
        

    def set_weights(self, matrix):        
        self.front.set_weights(matrix)        
        self.back.set_weights(matrix)
        self.head.set_weights(matrix)

    def cover_slow(self, matrix, center_x, center_y, part, configuration, weight_matrix):

        rows, cols = matrix.shape[:2]        
        #own_matrix = np.zeros((rows, cols, 1), np.uint8)
        #cv2.circle(own_matrix, (center_x, center_y), part.radius, (1), -1)                    

        sum = 0
        for x in range(center_x - part.radius, center_x + part.radius):
            for y in range(center_y - part.radius, center_y + part.radius):                
                if x < 0 or x >= cols:
                    continue
                if y < 0 or y >= rows:
                    continue
                if part.belongs_to(center_x, center_y, x, y):
                        sum += matrix[y, x] * weight_matrix[y, x]
                   
        return sum

    def cover(self, matrix, center_x, center_y, part, configuration, weight_matrix):

        mask = np.zeros((int(part.radius*2) + 1, int(part.radius*2) + 1), np.float)        
        cv2.circle(mask, (int(part.radius), int(part.radius)), int(part.radius), (1.0), -1)                    
        masked = np.multiply(mask, matrix[int(center_y - part.radius): int(center_y + part.radius) + 1, int(center_x - part.radius):int(center_x + part.radius) + 1])
        weighted =  np.multiply(masked, weight_matrix[int(center_y - part.radius): int(center_y + part.radius) + 1, int(center_x - part.radius):int(center_x + part.radius) + 1])
        
        return weighted.sum();        
        

    # recursive brute force "best fit"
    def do_fit(self, matrix, parts, configuration, weight_matrix):

        max = 0
        result_config = list(configuration)
        search_region = 6
        part, rest = parts[0], parts[1:]
        for x in xrange(int(part.center.x) - search_region / 2, int(part.center.x) + search_region / 2):
            for y in xrange(int(part.center.y) - search_region / 2, int(part.center.y) + search_region / 2):

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
                               
                new_config = list(configuration)
                new_config.append(self.PartConfiguration(x, y, part))                
                local_max = 0
                if rest:
                    new_weight_matrix = np.copy(weight_matrix)
                    cv2.circle(new_weight_matrix, (x, y), int(part.radius), (0.0), -1)                    
                    (local_max, local_max_config) = self.do_fit(matrix, rest, new_config, new_weight_matrix)
                else:
                    local_max_config = new_config
                
                local_max += self.cover(matrix, x, y, part, configuration, weight_matrix);
                if local_max > max:
                    max = local_max
                    result_config = local_max_config
                    
        return (max, result_config)
                
                
    
    def fit(self, matrix, weight_matrix):

        parts = [self.back, self.front, self.head]

        rows, cols = matrix.shape[:2]
        
        (max, configuration) = self.do_fit(matrix, parts, [], weight_matrix)
        
        for part_config in configuration:
            part_config.part.center.x = part_config.x
            part_config.part.center.y = part_config.y
            
            
class Animals:

    animals = []    
    
    def __init__(self, params):
        
        self.params = params        
    
    def add_animal(self, start_x, start_y, end_x, end_y):
        
        self.animals.append(Animal(self.params, start_x, start_y, end_x, end_y))
        
    def fit(self, matrix):

        rows, cols = matrix.shape[:2]
        
        for a in self.animals:
            
            weight_matrix = np.ones((rows, cols), np.float)
            
            for other in self.animals:                
                if other != a:
                    other.set_weights(weight_matrix)
                    
            a.fit(matrix, weight_matrix);    
            
class TrackingFlowElement:
                
    def __init__(self, filtered_image):
        self.filtered_image = filtered_image            

def calculate_scale_factor(frame_width, frame_height):
    width = tracking_resolution_width
    height = tracking_resolution_height
    k = float(frame_width) / frame_height
    if k > float(width) / height:
        return float(width) / frame_width
    else:
        return float(height) / frame_height

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



class Tracking:

    tracking_params = TrackingParams()
    
    def __init__(self, frame_width, frame_height):            
        self.tracking_params.scale_factor = calculate_scale_factor(frame_width, frame_height)        
        self.animals = Animals(self.tracking_params)

    def add_animal(self, start_x, start_y, end_x, end_y):
        
        self.animals.add_animal(start_x, start_y, end_x, end_y)
        
    def do_tracking(self, video_file, start_frame, tracking_flow, time_to_stop):

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

        if animal.head.center.x < animal.back.center.x:
            left_x = animal.head.center.x - animal.head.radius - tracking_border
            right_x = animal.back.center.x + animal.back.radius - tracking_border
        else:
            right_x = animal.head.center.x + animal.head.radius - tracking_border
            left_x = animal.back.center.x - animal.back.radius - tracking_border

        if animal.head.center.y < animal.back.center.y:
            top_y = animal.head.center.y - animal.head.radius - tracking_border
            bottom_y = animal.back.center.y + animal.back.radius - tracking_border
        else:
            bottom_y = animal.head.center.y + animal.head.radius - tracking_border
            top_y = animal.back.center.y - animal.back.radius - tracking_border
    
        roi = frame[top_y:bottom_y, left_x:right_x]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,50.)))
        #    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,200.)))

        roi_hist = cv2.calcHist([hsv_roi], [2], mask, [180], [0,180] )
        
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            
        while(not time_to_stop.isSet()):
   
           frame = resize(frame)

           border = tracking_border   
           frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, (0, 0, 0))
  
           hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           dst = cv2.calcBackProject([hsv], [2], roi_hist, [0,180], 1)  
    
           rows, cols = dst.shape[:2]
    
           self.animals.fit(dst)
                    
           tracking_flow_element = TrackingFlowElement(dst)       
           tracking_flow.put(tracking_flow_element)

           # read the next frame
           ret, frame = video.read()    

           if ret == False:
               break
       
             

        cv2.destroyAllWindows()
  
    
    
