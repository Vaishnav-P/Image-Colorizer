from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar
from tkinter import ttk
import test2_b as Colorizer
import threading 
import time

# class worker(threading.Thread):
# 	def __init__(self,variable,**kwargs):
# 		super().__init__(**kwargs)
# 		self.variable = variable

# 	def run(self):
# 		for i in range(101):
# 			self.variable.set(i)
# 			time.sleep(1)
# 		self.variable.set(0)	

# def startWorker():
# 	worker(progress).start()
def increment(*args):
	for i in range(100):
		bar['value'] = i+1
		root.update()
		time.sleep(0.1)


root = Tk()
root.title('Image Colorizer')
root.geometry('580x200')

global saveName

def browse_button():
    global folder_path
    global filename
    filename = filedialog.askopenfilename()
    e.insert(0, filename)


def ok_button():
   
    saveName = e1.get()
    completed=False
    if saveName =='':
        saveName = 'default'
    increment()
    completed = Colorizer.colorize(filename,saveName)
    if completed == True:
       messagebox.showinfo(title="Image Colorizer",message="Image colorized")
    else:
    	messagebox.showerror(title="Image Colorizer",message="An Error has occured")


frame = LabelFrame(root)
frame.grid(row=10, column=3)


lbl2 = Label(text=" ").grid(row=0,column=1)
e = Entry(root, borderwidth=2, width=40, bg="white")
e.grid(row=2, column=1)


folder_path = StringVar()
Label(text='Save As:').grid(row=3,column=0)
lbl1 = Label(master=root, text="Select:")
lbl1.grid(row=2, column=0)
button1 = Button(root, text="Browse", command=browse_button, anchor=W)
button1.grid(row=2, column=2, padx=10)
Label(text='Save As:').grid(row=3,column=0)
e1 = Entry(root,borderwidth=2,width=40,bg="white")
e1.grid(row=3,column=1)

button2 = Button(root, text="OK", command=ok_button)
button2.grid(row=3, column=2, pady=10)




style = ttk.Style()
style.theme_use('default')
style.configure("grey.Horizontal.TProgressbar", background='green')
bar = Progressbar(root, length=280, style='grey.Horizontal.TProgressbar',mode="determinate",maximum=100)
bar.grid(row=7,column=0,columnspan=5,pady=10)

root.mainloop()