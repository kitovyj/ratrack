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
