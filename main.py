import tkinter as tk
from PIL import ImageTk
import PIL.Image
import cv2
from tkinter import Frame, Label, Button, YES, LEFT, BOTH
from tkinter.messagebox import showinfo, askyesno
import threading
import dlib_utils

import numpy as np
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing import image


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

        # init gaze direction thread
        self.thread_g = threading.Thread(target=self.gaze_direction_thread)
        self.frame_queue = np.empty((0,128,32,3))

        #self.set_window_size()

    def set_window_size(self):
        screen_width = root.winfo_screenwidth() - 300
        screen_height = root.winfo_screenheight() - 300
        root.geometry('%sx%s+%s+%s' % (screen_width, screen_height, 0, 0))  #center window on desktop

    def popup_hello(self):  # for command
        showinfo("Hello", ">_<")

    def popup_hello_event(self, event):  # for bind
        showinfo("Hello", ">_<")
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
        self.thread_g.start()
        self.status['text'] = 'Using webcam'
    
    def load_model(self, event):
        self.thread_m.start()
        #self.thread_g.start()

    def load_model_thread(self):
        print('loading model...')
        #model_path = r'D:\DL\model\two_eyes_gaze_keras_resnet50_model.h5'
        model_path = r'D:\DL\model\20190419\two_eyes_gaze_keras_resnet50_model.json'
        model_weight_path = r'D:\DL\model\20190419\two_eyes_gaze_keras_resnet50_model.h5'
        with open(model_path, 'r') as file:
            model_json = file.read()
            self.model = model_from_json(model_json)
        self.model.load_weights(model_weight_path)
        self.model.predict(np.zeros((1, 128, 32, 3)))  # before using model, must predict once to avoid 'Tensor is not an element of this graph.'
        self.predict['text'] = 'Using model'
        print('loading completed!')

    def webcam_thread(self):
        print('start webcam_thread')
        # 設定影像的尺寸大小
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        
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
                    x = np.swapaxes(x,1,2)   # swap from (1, 32, 128, 3) to (1, 128, 32, 3)
                    self.frame_queue = np.vstack([self.frame_queue,x])
                else:
                    self.frame_queue = np.empty((0,128,32,3))


    def gaze_direction_thread(self):
        cls_list = ['LeftUp', 'Up', 'RightUp',
                    'Left', 'Center', 'Right',
                    'LeftDown', 'Down', 'RightDown']

        print('start gaze_direction_thread')
        while(True):
            if self.model is not None:
                if self.frame_queue.shape[0] >= 6:  # each 6 frame, predict a result
                    print('There are', self.frame_queue.shape[0], 'frames')
                    # gaze estimation
                    frames = self.frame_queue[:6]
                    self.frame_queue = np.delete(self.frame_queue, [0,1,2,3,4,5], axis=0)
                    predicts = self.model.predict(frames)
                    predict = predicts[0] + predicts[1] + predicts[2] + predicts[3] + predicts[4] + predicts[5]
                    top_class = np.argmax(predict)
                    
                    # change color & text
                    ans = str(cls_list[top_class]) + ' : ' + str(predict[top_class])
                    self.predict['text'] = ans
                    print(predict)
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

