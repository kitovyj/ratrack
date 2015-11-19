import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
from PIL import Image, ImageTk

class BodyPart:
        
    class Center:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
    def __init__(self, x, y, radius):
        self.center = self.Center(x, y)
        self.radius = radius
    
    def draw(self, matrix):        
        cv2.circle(matrix, (self.center.x, self.center.y), self.radius, (255, 255, 255))

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
        for x in range(self.center.x - self.radius, self.center.x + self.radius):
            for y in range(self.center.y - self.radius, self.center.y + self.radius):
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
        
        

class Animal:
    
    class PartConfiguration:
        def __init__(self, x, y, part):
            self.part = part
            self.x = x;
            self.y = y;        

    def __init__(self, front_x, front_y, front_radius, back_x, back_y, back_radius):
        self.front = BodyPart(front_x, front_y, front_radius)
        self.back = BodyPart(back_x, back_y, back_radius)
        
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
                
        self.head = BodyPart(int(head_x), int(head_y), head_radius)
        
    
    def shift(self, matrix):        
        self.front.shift(matrix)
        self.back.shift(matrix)
            
    def draw(self, matrix):        
        self.front.draw(matrix)        
        self.back.draw(matrix)
        self.head.draw(matrix)

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

        mask = np.zeros((part.radius*2 + 1, part.radius*2 + 1), np.float)        
        cv2.circle(mask, (part.radius, part.radius), part.radius, (1.0), -1)                    
        masked = np.multiply(mask, matrix[center_y - part.radius: center_y + part.radius + 1, center_x - part.radius:center_x + part.radius + 1])
        weighted =  np.multiply(masked, weight_matrix[center_y - part.radius: center_y + part.radius + 1, center_x - part.radius:center_x + part.radius + 1])
        
        return weighted.sum();        
        

    # recursive brute force "best fit"
    def do_fit(self, matrix, parts, configuration, weight_matrix):

        max = 0
        result_config = list(configuration)
        search_region = 6
        part, rest = parts[0], parts[1:]
        for x in range(part.center.x - search_region / 2, part.center.x + search_region / 2):
            for y in range(part.center.y - search_region / 2, part.center.y + search_region / 2):

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
                    cv2.circle(new_weight_matrix, (x, y), part.radius, (0.0), -1)                    
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
    
    def add_animal(self, front_x, front_y, front_radius, back_x, back_y, back_radius):
        
        self.animals.append(Animal(front_x, front_y, front_radius, back_x, back_y, back_radius))
        
    def draw(self, canvas):

        for a in self.animals:
            a.draw(canvas)
        
    def fit(self, matrix):

        rows, cols = matrix.shape[:2]
        
        for a in self.animals:
            
            weight_matrix = np.ones((rows, cols), np.float)
            
            for other in self.animals:                
                if other != a:
                    other.set_weights(weight_matrix)
                    
            a.fit(matrix, weight_matrix);    
            
            
def do_tracking(frames, time_to_stop):
    
    cap = cv2.VideoCapture('videotest.avi')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 190)

    # take first frame of the video
    ret, frame = cap.read()

    if not ret:
        print('can\'t read the video')
        sys.exit()
        
    animals = Animals()

    border = 20
    animals.add_animal(border + 133, border + 34, 5, border + 146, border + 34, 7)
    animals.add_animal(border + 103, border + 74, 5, border + 117, border + 78, 7)

    # setup initial location of window
    r, h, c, w = 20, 30, 115, 35  # simply hardcoded the values
    track_window = (c, r, w, h)
 
    initial_areas = [ ( 140, 23, 65, 25 ), ( 140, 23, 65, 25 ) ]
    
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((255.,255.,50.)))

    roi_hist = cv2.calcHist([hsv_roi], [2], mask, [180], [0,180] )

    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    while(not time_to_stop.isSet()):
   
       ret, frame = cap.read()    

       if ret == False:
           break
   
       frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_CONSTANT, (0, 0, 0))
  
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       dst = cv2.calcBackProject([hsv], [2], roi_hist, [0,180], 1)  
    
       rows, cols = dst.shape[:2]
    
       animals.fit(dst)
       animals.draw(dst)

       dst2x = cv2.resize(dst, (0, 0), fx = 2.0, fy = 2.0)
         
       animals.draw(frame)

       frame2x = cv2.resize(frame, (0, 0), fx = 2.0, fy = 2.0)
       
       frames.put(frame2x)
             

    cv2.destroyAllWindows()
    cap.release()    
    
    
