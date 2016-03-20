import math

class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    @staticmethod
    def from_tuple(t):
        return Point(t[0], t[1])
    def copy(self):
        return Point(self.x, self.y)
    def clone(self):
        return self.copy()
    def as_int_tuple(self):
        return (int(self.x), int(self.y))
    def add(self, p):
        self.x += p.x
        self.y += p.y
        return self        
    def difference(self, p):
        return Point(self.x - p.x, self.y - p.y)
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

def point_along_a_line_p(p1, p2, distance):
    (x, y) = point_along_a_line(p1.x, p1.y, p2.x, p2.y, distance)
    return Point(x, y)

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
    
def point_along_a_perpendicular_p(s, e, p, distance):
    return point_along_a_perpendicular(s.x, s.y, e.x, e.y, p.x, p.y, distance)

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

def distance_p(p1, p2):
    return distance(p1.x, p1.y, p2.x, p2.y)
    
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

def pcosine(p1, p2, p3):
    return cosine(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
    
def sgn(x):
    if x < 0:
        return -1
    else:
        return 1
    
def intersection_with_circle(p1, p2, center, radius):    
    p1s = p1.difference(center)
    p2s = p2.difference(center)
    # http://mathworld.wolfram.com/Circle-LineIntersection.html
    dx = p2s.x - p1s.x
    dy = p2s.y - p1s.y
    dr = math.sqrt(dx**2 + dy**2)
    D = p1s.x*p2s.y - p2s.x*p1s.y
    r = radius
    det = (r**2) * (dr**2) - (D**2)
    if det <= 0:
        return (None, None)        
    else:
        p1 = Point()
        p2 = Point()                        
        p1.x = (D * dy + sgn(dy) * dx * math.sqrt(det)) / (dr ** 2)
        p2.x = (D * dy - sgn(dy) * dx * math.sqrt(det)) / (dr ** 2)
        p1.y = (- D * dx + abs(dy) * math.sqrt(det)) / (dr ** 2)
        p2.y = (- D * dx - abs(dy) * math.sqrt(det)) / (dr ** 2)
        return (p1.add(center), p2.add(center))

def angle(x1, y1, pivot_x, pivot_y, x2, y2):
    x1 -= pivot_x
    y1 -= pivot_y
    x2 -= pivot_x
    y2 -= pivot_y
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    if det == 0 and dot == 0:
        return 0
    return math.atan2(det, dot)

# anticlockwise (clockwise if inverted y)

def rotate_p(point, pivot, angle):
    s = math.sin(angle);
    c = math.cos(angle);
    p = point.difference(pivot);
    x = p.x * c - p.y * s + pivot.x
    y = p.x * s + p.y * c + pivot.y
    return (x, y)


        
