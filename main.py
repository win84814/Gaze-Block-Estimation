import tkinter as tk
from PIL import ImageTk
import PIL.Image
import cv2
from tkinter import Frame, Label, Button, YES, LEFT, BOTH
from tkinter.messagebox import showinfo, askyesno
import threading


class Application(Frame):
    def __init__(self, master):
        
        # init main window 
        master.title("Gaze Block Estimation")
        master.resizable(False, False)

        # init left grid (for gaze blocks)
        frame1 = Frame(master)
        self.nine_grid = [[0 for x in range(3)] for x in range(3)] 
        content = [['nw', 'n', 'ne'], ['w', 'c', 'e'], ['sw', 's', 'se']]
        for row in range(3):
            for column in range(3):
                #Button(frame1,width=50,height=10, text=content[row][column]).grid(row = row, column = column)
                self.nine_grid[row][column] = Button(frame1, 
                                                     width=50, 
                                                     height=10, 
                                                     text=content[row][column])
                self.nine_grid[row][column].grid(row=row, column=column)
        frame1.pack(side=LEFT, fill=BOTH, expand=YES)

        self.nine_grid[1][1].bind("<Button-1>", self.popup_hello_event)

        # init right grid (for control buttons and webcam)
        frame2 = Frame(master)
        self.connect_to_camera_btn = Button(frame2, text='Connect')
        self.connect_to_camera_btn.bind("<Button-1>", self.open_camera)
        self.connect_to_camera_btn.pack()
        self.status = Label(frame2, text="Can't find camera")
        self.status.pack()
        self.predict = Label(frame2, text="Predict : center")
        self.predict.pack()

        # init webcam logo
        image = cv2.imread(r'D:\DL\code\Gaze-Block-Estimation\webcam.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.webcam = Label(frame2, image=image)
        self.webcam.image = image
        self.webcam.place(width=200, height=200)
        self.webcam.pack()
        frame2.pack(side=LEFT, padx=10)
        
        # init webcam object & thread
        self.thread_w = threading.Thread(target=self.webcam_thread)
        self.thread_w_signal = False
        self.capture = cv2.VideoCapture(0)

        #self.set_window_size()

    def set_window_size(self):
        screen_width = root.winfo_screenwidth() - 300
        screen_height = root.winfo_screenheight() - 300
        root.geometry('%sx%s+%s+%s' % (screen_width, screen_height, 0, 0))   #center window on desktop

    def popup_hello(self):  # for command
        showinfo("Hello", ">_<")

    def popup_hello_event(self, event):  # for bind
        showinfo("Hello", ">_<")
    
    def open_camera(self, event):
        self.thread_w_signal = True
        self.thread_w.start()
    
    def webcam_thread(self):

        # 設定影像的尺寸大小
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

        while(self.thread_w_signal):
            ret, frame = self.capture.read()
            tkframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tkframe = PIL.Image.fromarray(tkframe)
            tkframe = ImageTk.PhotoImage(tkframe)
            self.webcam.imgtk = tkframe
            self.webcam.config(image=tkframe)

    def on_closing(self):
        ans = askyesno(title='Quit', message='Do you want to quit?')
        if ans:
            self.thread_w_signal = False
            root.destroy()
        else:
            return

root = tk.Tk()
app = Application(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()

