from tkinter import *
from tkinter import filedialog
import test2_b as Colorizer

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
    backup = var_backup.get()
    saveName = e1.get()
    completed=False
    if saveName =='':
        saveName = 'default'
    completed = Colorizer.colorize(filename,saveName)
    if completed == True:
        msg.set("Image Colorized and Saved in Installation Folder")


frame = LabelFrame(root)
frame.grid(row=10, column=3)


lbl2 = Label(text="Colorizer").grid(row=0,column=1)
e = Entry(root, borderwidth=2, width=40, bg="white")
e.grid(row=2, column=1)


folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=2, column=0)
button1 = Button(root, text="Browse", command=browse_button, anchor=W)
button1.grid(row=2, column=2, padx=10)

Label(text='Save As:').grid(row=3,column=0)
e1 = Entry(root,borderwidth=2,width=40,bg="white")
e1.grid(row=3,column=1)

button2 = Button(root, text="OK", command=ok_button)
button2.grid(row=3, column=2, pady=10)

msg = StringVar()

Label(textvariable=msg).grid(row = 5,column=1)


var_backup = BooleanVar()
c1 = Checkbutton(root, text='Backup', variable=var_backup,
                 onvalue=True, offvalue=False, anchor=E)
c1.deselect()
c1.grid(row=4, column=0, columnspan=5)

root.mainloop()