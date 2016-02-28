import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def copy(self):
        return Point(self.x, self.y)
    def clone(self):
        return self.copy()
    def as_int_tuple(self):
        return (int(self.x), int(self.y))
    def scaled(self, factor, shift = 0):
        return Point(self.x * factor + shift, self.y * factor + shift)
    def as_int(self):
        return Point(int(self.x), int(self.y))

def point_along_a_line(start_x, start_y, end_x, end_y, distance):

    dx = end_x - start_x
    dy = end_y - start_y
    x = 0
    y = 0
    if dx != 0:
        k = float(dy) / float(dx)
        point_dx = math.sqrt(distance**2 / (1 + k**2))
        if dx < 0:
            point_dx *= -1
        point_dy = point_dx * k        
        x = start_x + point_dx
        y = start_y + point_dy        
    else:
        x = start_x
        if dy < 0:
            y = start_y - distance
        else:
            y = start_y + distance
    return (x, y)                 

def point_along_a_perpendicular(start_x, start_y, end_x, end_y, p_start_x, p_start_y, distance):
    dx = end_x - start_x
    dy = end_y - start_y
    x = 0
    y = 0
    if dx != 0:
        k = float(dy) / float(dx)
        point_dx = math.sqrt(distance**2 / (1 + k**2))
        if dx < 0:
            point_dx *= -1
        point_dy = point_dx * k  
        if distance < 0:
            point_dx *= -1
            point_dy *= -1            
        x = p_start_x - point_dy
        y = p_start_y + point_dx        
    else:
        y = p_start_y
        if dy < 0:
            x = p_start_x - distance
        else:
            x = p_start_x + distance
    return (x, y)                 

def point_along_a_line_eq(k, start_x, start_y, distance):
    x = 0
    y = 0
    if k != 0:
        point_dx = math.sqrt(distance**2 / (1 + k**2))
        if distance < 0:
            point_dx *= -1
        point_dy = point_dx * k        
        x = start_x + point_dx
        y = start_y + point_dy        
    else:
        x = start_x
        y = start_y + distance
    return (x, y)                 

def distance(start_x, start_y, end_x, end_y):
    return math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
    
def line_equation(start_x, start_y, end_x, end_y):
    dx = end_x - start_x
    k = 0
    if dx != 0:        
        k = float(end_y - start_y) / (end_x - start_x)
    return (k, start_y)

def cosine(x1, y1, x2, y2, x3, y3)                    :
    sx1 = x1 - x2
    sy1 = y1 - y2
    sx2 = x3 - x2
    sy2 = y3 - y2
    dot = sx1 * sx2 + sy1 * sy2
    a = math.sqrt(sx1**2 + sy1**2)
    b = math.sqrt(sx2**2 + sy2**2)
    prod = a * b
    if prod != 0:
        cos = dot / prod
        if cos > 1.0:
            cos = 1.0
        elif cos < -1.0:
            cos = -1.0
        return cos
    else:
        return 0        
    