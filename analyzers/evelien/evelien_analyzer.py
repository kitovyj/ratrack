from tracking import *
from geometry import *

import datetime

class EvelienAnalyzer:
    
    class CompartementStatistics:
        def __init__(self):
            self.distance = 0
            self.time = 0        

    comp_z = 0
    comp_b = 1
    
    current_compartement = None
    current_compartement_enter_time = None
    
    compartement_times = []
    trajectory = []
    
    compartment_time = [0., 0.]
    compartment_distance = [0., 0.]
    total_distance = 0
    
    max_bin_duration = 60.

    start_time = None
    
    class Bin:
        def __init__(self):
            self.duration = 0
            self.distance = 0
            # a list of CompartementStatistics
            self.statistics = []
            for c in [EvelienAnalyzer.comp_z, EvelienAnalyzer.comp_b]:
                self.statistics.append(EvelienAnalyzer.CompartementStatistics())
    
    bins = []
    
    current_position = None

    class Configuration:        
        def __init__(self):
            self.circle_center = geometry.Point(0, 0)
            self.circle_radius = 1        
            self.outer_circle_radius = 1
            self.area = 1. / 3                        
                    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    def analyze(self, tracking_data):

        p = tracking_data.positions

        # if no animals in the list
        if (p is None) or (not p):
            return       
            
        ap = p[0][1]
        
        # find min and max values

        max_val = -1
        min_val = -1
        max_i = 0
                
        for idx, v in enumerate(ap.backbone):
            if v.value > max_val:
                max_val = v.value
                max_i = idx
            if min_val == -1 or v.value < min_val:
                min_val = v.value
        
        v = ap.backbone[max_i]
                
        if self.current_position is None:

            self.current_position = v.center
            self.trajectory.append((0, self.current_position))
            self.bins.append(self.Bin())                        
            self.start_time = tracking_data.time
            self.current_compartement = self.compartement(v.center)
            self.current_compartement_enter_time = 0

            self.log_with_time(0, 'the animal enters ' + self.compartement_str(self.current_compartement))
            
            return
            
        else:
            
            d = geometry.pdistance(self.current_position, v.center)
            if d > 10:
                self.current_position = v.center
                self.trajectory.append((tracking_data.time - self.start_time, self.current_position))
                dist_travelled = d
                new_compartement = self.compartement(self.current_position)
            else:
                dist_travelled = 0
                self.trajectory.append((tracking_data.time - self.start_time, self.trajectory[-1][1].clone()))
                        
        prev = self.trajectory[-2][1]
        curr = self.trajectory[-1][1]                
        prev_t = self.trajectory[-2][0]
        curr_t = self.trajectory[-1][0]

        time_passed = curr_t - prev_t;
        
        entered_a_new_compartement = False

        if dist_travelled > 0:

            self.total_distance += dist_travelled
            
            if new_compartement != self.current_compartement:                       
            
                entered_a_new_compartement = True

                (p1, p2) = geometry.intersection_with_circle(prev, curr, self.config.circle_center, self.config.circle_radius)
                
                if p1 is None:

                    dist_in_new_comp = dist_travelled                   

                else:           
                    
                    if new_compartement == self.comp_z:
                        closest = prev
                    else:
                        closest = curr           
                        
                    if geometry.pdistance(closest, p1) > geometry.pdistance(closest, p2):
                        ip = p2
                    else:
                        ip = p1                        
                        
                    dist_in_new_comp = dist_travelled - geometry.pdistance(curr, ip)
                    
                dist_in_prev_comp = dist_travelled - dist_in_new_comp
                                
                crossing_time = prev_t + time_passed * dist_in_prev_comp / dist_travelled
            
                self.compartment_time[self.current_compartement] += crossing_time - self.current_compartement_enter_time
                self.compartment_distance[self.current_compartement] += dist_in_prev_comp                        
                self.compartment_distance[new_compartement] += dist_in_new_comp
                               
                self.log_with_time(crossing_time, 'the animal enters ' + self.compartement_str(new_compartement))
                
            else:
                
                self.compartment_distance[self.current_compartement] += dist_travelled
            
        # lets deal with bins now
                
        time_left = time_passed
                
        prev_time_point = prev_t
        
        last_compartement = self.current_compartement
        last_compartement_enter_time = self.current_compartement_enter_time
                                     
        while time_left > 0:
            
            # bin start time 
            t0 = (len(self.bins) - 1) * self.max_bin_duration
            # bin upper time limit
            t1 = t0 + self.max_bin_duration    

            time_left_in_bin = t1 - prev_time_point
                        
            # time goes to the bin
            if time_left_in_bin > time_passed:    
                bin_time = time_passed
            else:
                bin_time = time_left_in_bin

            bin = self.bins[-1]
                                        
            if dist_travelled > 0:
                                                
                # distance travelled during this bin interval
                bin_dist = dist_travelled * bin_time / time_passed
                bin.distance += bin_dist
                
                if not entered_a_new_compartement or t1 < crossing_time:                    
                
                    bin.statistics[self.current_compartement].distance += bin_dist                    
                        
                elif t0 < crossing_time and t1 > crossing_time:
                    
                    # distance travelled before to entering this bin
                    time_consumed = time_passed - time_left
                    bin_start_dist = dist_travelled * time_consumed / time_passed
                    bin_dist_in_prev_comp = dist_in_prev_comp - bin_start_dist                                                            
                    curr_comp_time = crossing_time - max(self.current_compartement_enter_time, t0)
                    
                    bin.statistics[self.current_compartement].distance += bin_dist_in_prev_comp
                    bin.statistics[self.current_compartement].time += curr_comp_time
                    bin.statistics[new_compartement].distance += bin_dist - bin_dist_in_prev_comp
                    last_compartement = new_compartement
                    last_compartement_enter_time = crossing_time
                    
                else:
                    
                    bin.statistics[new_compartement].distance += bin_dist

            time_left = time_left - bin_time

            prev_time_point = t1            
            
            if time_left > 0: 
                # 'close' bin
                last_compartment_time = min(t1 - last_compartement_enter_time, self.max_bin_duration)  
                bin.statistics[last_compartement].time += last_compartment_time
                bin.duration = self.max_bin_duration
                self.log_bin_statistics(bin, len(self.bins) - 1)
                self.bins.append(self.Bin())
                
                            
        if entered_a_new_compartement:                    
            self.current_compartement_enter_time = crossing_time
            self.current_compartement = new_compartement
                

    def log_bin_statistics(self, bin, bin_index):

        t0 = bin_index * self.max_bin_duration
        t1 = t0 + self.max_bin_duration    
                        
        self.logger.log('')        
        
        ts = '[' + self.time_str(t0) + ', ' + self.time_str(t1) + ']'

        self.logger.log(ts + ' bin data:')
        self.logger.log('   total distance: ' + str(bin.distance)) 
        for c in [self.comp_z, self.comp_b]:
            self.logger.log('   total time in ' + self.compartement_str(c) + ': ' + str(bin.statistics[c].time))
            self.logger.log('   total distance in ' + self.compartement_str(c) + ': ' + str(bin.statistics[c].distance))        
        self.logger.log('')
                                                            
    def on_finished(self):
                
        if self.trajectory:
            last_time = self.trajectory[-1][0]
            self.compartment_time[self.current_compartement] += last_time - self.current_compartement_enter_time
            last_bin = self.bins[-1]
            bin_t0 = self.start_time + (len(self.bins) - 1) * self.max_bin_duration
            last_bin.statistics[self.current_compartement].time += last_time - max(bin_t0, self.current_compartement_enter_time)
            last_bin.duration = last_time - bin_t0
            self.log_bin_statistics(last_bin, len(self.bins) - 1)
            total_time = last_time - self.start_time                        
            
        else:
            total_time = 0
            
        if not (self.logger is None):
            self.logger.log('total distance: ' + str(self.total_distance))
            self.logger.log('total time: ' + str(total_time))
            for c in [self.comp_z, self.comp_b]:
                self.logger.log('total time in ' + self.compartement_str(c) + ': ' + str(self.compartment_time[c]))
                self.logger.log('total distance in ' + self.compartement_str(c) + ': ' + str(self.compartment_distance[c]))
        
        
    def compartement(self, p):
        d = geometry.pdistance(p, self.config.circle_center)
        if d < self.config.circle_radius:
            return self.comp_z
        else:
            return self.comp_b
                                        
    def compartement_str(self, c):
        if c == self.comp_z:
            return 'Central compartment '
        else:
            return 'Boundary compartment'

    def log_with_time(self, time, message):
        self.logger.log('[' + self.time_str(time) + '] ' + message)
        
        
    def time_str(self, seconds):
        return str(datetime.timedelta(seconds = seconds)) 
        
        
    
    