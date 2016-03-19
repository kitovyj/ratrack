import sys
import numpy as np
import cv2
import math

class EvelienDecorator:
                    
    def __init__(self, config):        
        self.config = config        

    def decorate_after(self, analyzer, image, image_scale_factor):        
        
        white = (255, 255, 255)
        red = (255, 0, 0)            
        
        if not (self.config.circle_center is None):
            
            image_height, image_width = image.shape[:2]                        

            center = self.config.circle_center.scaled(image_scale_factor)
                         
            cross_size = 5                                    
                                    
            cv2.line(image, (int(center.x - cross_size), int(center.y)),
                     (int(center.x + cross_size), int(center.y)), white)
            cv2.line(image, (int(center.x), int(center.y - cross_size)),
                     (int(center.x), int(center.y + cross_size)), white)

            radius = int(self.config.circle_radius * image_scale_factor)
            

            #cv2.ellipse(image, center.as_int_tuple(), (radius, radius), int(0), int(90), white)

            color = white

            if (not (analyzer is None)):                
                if analyzer.current_compartement == analyzer.comp_z:
                    color = red
                    

            cv2.circle(image, center.as_int_tuple(), radius, color)     
        
        
    def decorate_before(self, analyzer, image, image_scale_factor):        

        if analyzer is None:
            return
            
        if not analyzer.trajectory:
            return

        prev = 0
        max_time = 15.
        for tp in reversed(analyzer.trajectory):
            p = tp[1]            
            if prev == 0:
                prev = p
                now = tp[0]
                continue
                        
            passed = now - tp[0]
            intensity = int(max(0, 255 * (max_time - passed) / max_time))            
            if intensity <= 0:
                break
            color = (intensity, intensity, intensity)            
            cv2.line(image, p.scaled(image_scale_factor).as_int_tuple(), prev.scaled(image_scale_factor).as_int_tuple(), color)                        
            prev = p
            
                        
    