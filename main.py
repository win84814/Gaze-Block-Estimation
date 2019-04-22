import tkinter as tk
from PIL import ImageTk
import PIL.Image
import cv2
from tkinter import Frame, Label, Button, YES, LEFT, BOTH
from tkinter.messagebox import showinfo, askyesno
import threading
import dlib_utils

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image


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
                                                     width=80, 
                                                     height=15, 
                                                     text=content[row][column])
                self.nine_grid[row][column].grid(row=row, column=column)
        self.nine_grid[1][1].bind("<Button-1>", self.popup_hello_event)
        frame1.pack(side=LEFT, fill=BOTH, expand=YES)

        # init right grid (for control buttons and webcam)
        frame2 = Frame(master)
        self.connect_to_camera_btn = Button(frame2, text='Connect')
        self.connect_to_camera_btn.bind("<Button-1>", self.open_camera)
        self.connect_to_camera_btn.pack()

        self.load_model_btn = Button(frame2, text='Load model')
        self.load_model_btn.bind("<Button-1>", self.load_model)
        self.load_model_btn.pack()

        self.status = Label(frame2, text="Can't find camera")
        self.status.pack()
        self.predict = Label(frame2, text="No model")
        self.predict.pack()

        # init webcam logo
        image = cv2.imread(r'D:\DL\code\Gaze-Block-Estimation\webcam.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.webcam = Label(frame2, image=image)
        self.webcam.image = image
        self.webcam.place(width=50, height=50)
        self.webcam.pack()
        frame2.pack(side=LEFT, padx=0)
        
        # init webcam object & thread
        self.thread_w = threading.Thread(target=self.webcam_thread)
        self.thread_w_signal = False
        self.capture = cv2.VideoCapture(0)

        # init model thread
        self.thread_m = threading.Thread(target=self.load_model_thread)
        self.model = None

        #self.set_window_size()

    def set_window_size(self):
        screen_width = root.winfo_screenwidth() - 300
        screen_height = root.winfo_screenheight() - 300
        root.geometry('%sx%s+%s+%s' % (screen_width, screen_height, 0, 0))  #center window on desktop

    def popup_hello(self):  # for command
        showinfo("Hello", ">_<")

    def popup_hello_event(self, event):  # for bind
        #showinfo("Hello", ">_<")
        #orig_color = self.nine_grid[0][0].cget("background")

        self.nine_grid[1][0].configure(bg="red")
        self.nine_grid[0][0].configure(bg="SystemButtonFace")
    
    def reset_nine_grid_color(self):
        for x in range(3):
            for y in range(3):
                self.nine_grid[x][y].configure(bg="SystemButtonFace")

    def popup_qq_event(self):
        showinfo("QQ", "QAQ")

    def open_camera(self, event):
        self.thread_w_signal = True
        self.thread_w.start()
        self.status['text'] = 'Using webcam'
    
    def load_model(self, event):
        self.thread_m.start()

    def load_model_thread(self):
        print('loading model...')
        model_path = r'D:\DL\model\two_eyes_gaze_keras_resnet50_model.h5'
        self.model = load_model(model_path)
        
        self.model.predict(np.zeros((1, 32, 128, 3)))  # before using model, must predict once to avoid 'Tensor is not an element of this graph.'

        self.predict['text'] = 'Using model'
        print('loading completed!')

    def webcam_thread(self):
        cls_list = ['LeftUp', 'Up', 'RightUp',
                    'Left', 'Center', 'Right',
                    'LeftDown', 'Down', 'RightDown']

        # 設定影像的尺寸大小
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 刷新縮圖
        while(self.thread_w_signal):
            ret, frame = self.capture.read()  # frame is BGR
            tkframe = dlib_utils.find_face_and_eyes(frame)
            tkframe = cv2.cvtColor(tkframe, cv2.COLOR_BGR2RGB)  # transfer BGR to RGB
            tkframe = PIL.Image.fromarray(tkframe)
            tkframe = ImageTk.PhotoImage(tkframe)
            self.webcam.imgtk = tkframe
            self.webcam.config(image=tkframe)

            # predict
            if self.model is not None:
                catch, two_eyes = dlib_utils.crop_two_eyes(frame) 
                
                self.reset_nine_grid_color()
                # avoid crop nothing
                if catch:
                    two_eyes = cv2.cvtColor(two_eyes, cv2.COLOR_BGR2RGB)  # transfer BGR to RGB
                    x = image.img_to_array(two_eyes)
                    x = np.expand_dims(x, axis=0)  # shape(1, 32, 128, 3)
                    x /= 255  # normalize
                    pred = self.model.predict(x)[0]
                    top_inds = pred.argsort()[::-1][:5]
                    top_class = top_inds[0]
                    ans = str(cls_list[top_class]) + ' : ' + str(pred[top_class])
                    self.predict['text'] = ans
                    print(ans)
                    self.nine_grid[int(top_class / 3)][top_class % 3].configure(bg="red")
                
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

