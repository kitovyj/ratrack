import sys
import numpy as np
import cv2
import math
import threading
import Queue

import Tkinter as Tk
from PIL import Image, ImageTk

from tracking import do_tracking
                        

frames = Queue.Queue()    
    
time_to_stop = threading.Event()

tracking_thread = threading.Thread(target = do_tracking, args = (frames, time_to_stop))

tracking_thread.start()

root = Tk.Tk()

w = Tk.Label(root, text="Hello, world!")
w.pack()

def quit():
    time_to_stop.set()    
    tracking_thread.join()
    root.quit()

button = Tk.Button(root, text = "Quit", command = quit)
button.pack(side = Tk.LEFT)

imglabel = Tk.Label(root)
imglabel.pack()

def poll_queue():
    
    if not frames.empty():
        frame = frames.get()

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image = img) 
   
        imglabel.configure(image = imgtk)
        imglabel.image = imgtk        
    
    root.after(100, poll_queue)

poll_queue()
root.mainloop()
root.destroy()

