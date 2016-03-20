from geometry import *
          
class Vertebra:
    def __init__(self, x, y, value = 0):
        self.center = geometry.Point(x, y)            
        self.value = value
    def clone(self):
        return Vertebra(self.center.x, self.center.y, self.value)
                                    
    