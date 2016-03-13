import tracking
import geometry

class EvelienAnalyzer:

    comp_unknown = 0
    comp_z = 1
    comp_1 = 2
    comp_2 = 3
    comp_3 = 4
    comp_4 = 5
    
    current_compartement = comp_unknown        
    current_compartement_enter_time = 0.0
    new_compartement = comp_unknown        
    new_compartement_enter_time = 0.0
    last_data_time = 0
    
    compartement_times = []
    trajectory = []
    
    compartment_time = [0., 0., 0., 0., 0.]
    compartment_distance = [0., 0., 0., 0., 0.]
    
    current_position = None

    class Configuration:        
        def __init__(self):
            self.circle_center = geometry.Point(0, 0)
            self.circle_radius = 1        
                    
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
        val_delta = max_val - min_val
        
        v = ap.backbone[max_i]
        
        self.last_data_time = tracking_data.time
        
        min_compartement_time = 0.5        
        
        if self.current_position is None:
            self.current_position = v.center
            self.trajectory.append((tracking_data.time, self.current_position))
        else:
            d = geometry.pdistance(self.current_position, v.center)
            if d > 4:
                self.current_position = v.center
                self.trajectory.append((tracking_data.time, self.current_position))                
                self.compartment_distance[self.current_compartement - 1] = self.compartment_distance[self.current_compartement - 1] + d
        
        if self.current_compartement == self.comp_unknown:
            self.current_compartement = self.compartement(v.center)
            self.current_compartement_enter_time = tracking_data.time
            
            if not (self.logger is None):
                self.logger.log('Animal enters ' + self.compartement_str(self.current_compartement))
            
        else:
            c = self.compartement(v.center)
            if c == self.current_compartement:
                self.new_compartement = self.comp_unknown                    
            elif self.new_compartement == c:
                if tracking_data.time - self.new_compartement_enter_time >= min_compartement_time:
                    self.compartement_times.append((self.current_compartement, self.current_compartement_enter_time, self.new_compartement_enter_time))

                    self.compartment_time[self.current_compartement - 1] = self.compartment_time[self.current_compartement - 1] + self.new_compartement_enter_time - self.current_compartement_enter_time

                    self.current_compartement = c
                    self.current_compartement_enter_time = self.new_compartement_enter_time
                    self.new_compartement = self.comp_unknown
                    
                    if not (self.logger is None):
                        self.logger.log('Animal enters ' + self.compartement_str(self.current_compartement))
                        
            else:
                self.new_compartement = c
                self.new_compartement_enter_time = tracking_data.time                                    
                
    def on_finished(self):
        self.compartment_time[self.current_compartement - 1] = self.compartment_time[self.current_compartement - 1] + self.last_data_time - self.current_compartement_enter_time
        if not (self.logger is None):
            for c in [self.comp_z, self.comp_1, self.comp_2, self.comp_3, self.comp_4]:
                self.logger.log('Time in ' + self.compartement_str(c) + ': ' + str(self.compartment_time[c - 1]))
                self.logger.log('Distance in ' + self.compartement_str(c) + ': ' + str(self.compartment_distance[c - 1]))
        
        
    def compartement(self, p):
        d = geometry.pdistance(p, self.config.circle_center)
        if d < self.config.circle_radius:
            return self.comp_z
        if p.x < self.config.circle_center.x:            
            if p.y < self.config.circle_center.y:
                return self.comp_1
            else:
                return self.comp_3
        else:
            if p.y < self.config.circle_center.y:
                return self.comp_2
            else:
                return self.comp_4
                                        
    def compartement_str(self, c):
        if c == self.comp_1:
            return 'Compartment 1'
        elif c == self.comp_2:
            return 'Compartment 2'
        elif c == self.comp_3:
            return 'Compartment 3'
        elif c == self.comp_4:
            return 'Compartment 4'
        else:
            return 'Central compartment '
        
        
        
        
    
    