import Tkinter as Tk

# tkinter layout management : http://zetcode.com/gui/tkinter/layout/                        

# helper code allowing setting control size in pixels

def create_button(root, ptext, pcommand, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    button = Tk.Button(f, text = ptext, command = pcommand)    
    button.pack(fill = Tk.BOTH, expand = 1)
    return button

def create_check(root, ptext, initial, pcommand, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    var = Tk.IntVar()
    var.set(initial)
    control = Tk.Checkbutton(f, text = ptext, command = pcommand, variable = var)    
    control.pack(fill = Tk.Y, expand = 1, anchor = Tk.W)
    control.var = var
    return control

def create_listbox(root, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    lb = Tk.Listbox(f)    
    lb.pack(fill = Tk.BOTH, expand = 1)
    return (f, lb)

def create_radio(root, ptext, var, val, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    control = Tk.Radiobutton(f, text = ptext, variable = var, value = val)    
    control.pack(fill = Tk.Y, expand = 1, anchor = Tk.W)
    return control

def create_label(root, ptext, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    control = Tk.Label(f, text = ptext)    
    control.pack(fill = Tk.BOTH, expand = 1)
    return control

def create_entry(root, val, px, py, pw, ph):
    f = Tk.Frame(root, height = ph, width = pw)
    f.pack_propagate(0) # don't shrink
    f.place(x = px, y = py)
    var = Tk.StringVar()
    var.set(val)
    control = Tk.Entry(f, textvariable = var)    
    control.pack(fill = Tk.Y, expand = 1, anchor = Tk.W)
    control.var = var
    return control
