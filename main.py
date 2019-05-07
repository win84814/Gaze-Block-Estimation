import tkinter as tk
from PIL import ImageTk, Image
import cv2
from tkinter import Frame, Label, Button, YES, LEFT, BOTH, X
from tkinter.messagebox import showinfo, askyesno
import threading
import dlib_utils
import numpy as np
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing import image
import time

#img_width, img_height = 128, 32
#model_path = r'D:\DL\model\20190424\two_eyes_gaze_keras_resnet50_model.json'
#model_weight_path = r'D:\DL\model\20190424\two_eyes_gaze_keras_resnet50_model.h5'

img_width, img_height = 128, 32
model_path = r'D:\DL\code\weight\20190429\two_eyes_gaze_keras_resnet50_model.json'
model_weight_path = r'D:\DL\code\weight\20190429\two_eyes_gaze_keras_resnet50_model.h5'

nine_grid_button_width = 70  # 91 #80
nine_grid_button_height = 20  # 25 #20

mode_list = ['Demo for big nine grid', 'Demo for big camera', 'Collect data']
mode = mode_list[2]


class Application(Frame):
    def __init__(self, master):
        
        # init main window 
        master.title("Gaze Block Estimation")
        master.resizable(False, False)
        self.frame1 = Frame(master)
        self.frame2 = Frame(master)

        # init left grid (for gaze blocks)
        self.nine_grid = [[0 for x in range(3)] for x in range(3)] 
        content = [['nw', 'n', 'ne'], ['w', 'c', 'e'], ['sw', 's', 'se']]
        for row in range(3):
            for column in range(3):
                #Button(frame1,width=50,height=10, text=content[row][column]).grid(row = row, column = column)
                self.nine_grid[row][column] = Button(self.frame1, 
                                                     width=nine_grid_button_width, 
                                                     height=nine_grid_button_height, 
                                                     text=content[row][column])
                self.nine_grid[row][column].grid(row=row, column=column)
        #self.frame1.pack(side=LEFT, fill=BOTH, expand=YES)

        self.set_window_size(mode)

        # init right grid (for control buttons and webcam)
        #self.frame2 = Frame(master)
        self.connect_to_camera_btn = Button(self.frame2, text='Connect')
        self.connect_to_camera_btn.bind("<Button-1>", self.open_camera)
        self.connect_to_camera_btn.pack()

        self.load_model_btn = Button(self.frame2, text='Load model')
        self.load_model_btn.bind("<Button-1>", self.load_model)
        self.load_model_btn.pack()

        self.status = Label(self.frame2, text="Can't find camera")
        self.status.pack()
        self.predict = Label(self.frame2, text="No model")
        self.predict.pack()

        # init webcam logo
        image = cv2.imread(r'D:\DL\code\Gaze-Block-Estimation\webcam.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.webcam = Label(self.frame2, image=image)
        self.webcam.image = image
        self.webcam.place(width=50, height=50)
        self.webcam.pack()
        # self.frame2.pack(side=LEFT, padx=0) ### comment for disable right side
        
        # init webcam object & thread
        self.thread_w = threading.Thread(target=self.webcam_thread)
        self.thread_w_signal = False
        self.capture = cv2.VideoCapture(0)

        # init model thread
        self.thread_m = threading.Thread(target=self.load_model_thread)
        self.model = None

        # init test thread for get data
        self.thread_c = threading.Thread(target=self.cycle_thread)

        # init frame counter
        self.frame_counter = np.full((3), -1)
        
    def set_window_size(self, select_mode):
        if select_mode == mode_list[0]:
            for row in range(3):
                for col in range(3):
                    self.nine_grid[row][col].config(height=20, width=80)
            self.frame1.pack(side=LEFT, fill=BOTH, expand=YES)
            self.frame2.pack(side=LEFT, padx=0)  # comment for disable right side

        elif select_mode == mode_list[1]:
            print('nothing')
        elif select_mode == mode_list[2]:
            # nine_grid_button_width =   90
            # nine_grid_button_height =  25
            self.frame1.pack(fill=X)
            root.attributes('-fullscreen', True)
            root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
            for row in range(3):
                for col in range(3):
                    # self.nine_grid[row][col].bind("<Button-1>", self.popup_hello_event)
                    self.nine_grid[row][col].config(height=25, width=90)
                    self.nine_grid[row][col].bind("<Button-1>", lambda event, r=row, c=col: self.click_nine_grid_button(r, c))
            # self.thread_s.start()

        # screen_width = root.winfo_screenwidth() - 300
        # screen_height = root.winfo_screenheight() - 300
        # root.geometry('%sx%s+%s+%s' % (screen_width, screen_height, 0, 0))  # center window on desktop

    def popup_hello(self):  # for command
        showinfo("Hello", ">_<")

    def click_nine_grid_button(self, r, c):
        # show red one
        self.reset_nine_grid_color()
        self.set_nine_grid_color(r, c)

        # init camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # init save path
        frame_count = 0
        save_path = ''
        while(frame_count < 100):
            ret, frame = self.capture.read() 

            catch, tkframe, two_eyes = dlib_utils.find_face_and_crop_two_eyes(frame, img_width, img_height)

            frame_count += 1
        
        # reset red one
        self.reset_nine_grid_color()
    
    def set_nine_grid_color(self, row, col):
        self.nine_grid[row][col].configure(bg="red")

    def reset_nine_grid_color(self):
        for row in range(3):
            for col in range(3):
                self.nine_grid[row][col].configure(bg="SystemButtonFace")

    def popup_qq_event(self):
        showinfo("QQ", "QAQ")

    def open_camera(self, event):
        self.thread_w_signal = True
        self.thread_w.start()
        self.status['text'] = 'Using webcam'
        # self.thread_t.start()

    def load_model(self, event):
        self.thread_m.start()

    def load_model_thread(self):
        print('loading model...')
        with open(model_path, 'r') as file:
            model_json = file.read()
            self.model = model_from_json(model_json)
        self.model.load_weights(model_weight_path)
        self.model.predict(np.zeros((1, img_height, img_width, 3)))  # before using model, must predict once to avoid 'Tensor is not an element of this graph.'
        self.predict['text'] = 'Using model'
        print('loading completed!')

    def webcam_thread(self):
        print('start webcam_thread')
        # 設定影像的尺寸大小
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 刷新縮圖
        while(self.thread_w_signal):
            start = time.time()
            ret, frame = self.capture.read() 

            catch, tkframe, two_eyes = dlib_utils.find_face_and_crop_two_eyes(frame, img_width, img_height)

            tkframe = cv2.cvtColor(tkframe, cv2.COLOR_BGR2RGB)  # transfer BGR to RGB
            tkframe = Image.fromarray(tkframe)
            tkframe = ImageTk.PhotoImage(tkframe)
            self.webcam.imgtk = tkframe
            self.webcam.config(image=tkframe)

            # predict
            if self.model is not None:
                # avoid crop nothing

                if catch:
                    # preprocessing
                    x = cv2.cvtColor(two_eyes, cv2.COLOR_BGR2RGB)  # transfer BGR to RGB
                    x = np.expand_dims(x, axis=0)  # shape(1, 32, 128, 3)
                    x = x.astype(np.float32)  # tranfer uint8 to float32
                    x /= 255  # normalize

                    # predict gaze direction
                    predict = self.model.predict(x)[0]
                    top_class = np.argmax(predict)

                    # count estimation
                    self.frame_counter[1] = top_class
                    if self.frame_counter[1] == self.frame_counter[0]:
                        self.frame_counter[2] += 1
                    else:
                        self.frame_counter[2] = 0\
                    
                    # change color
                    if self.frame_counter[2] >= 7:
                        self.reset_nine_grid_color()
                        self.set_nine_grid_color(int(top_class / 3), top_class % 3)
                        self.frame_counter[2] -= 8
                    
                    print(self.frame_counter)
                    self.frame_counter[0] = self.frame_counter[1]

            end = time.time()
            seconds = end - start
            print('fps', (1/seconds))

    def cycle_thread(self):    
        start = time.time()
        pikapika = -1
        while(True):
            now = int(time.time() - start)
            print(now, 's')
            if now % 18 != pikapika:
                self.reset_nine_grid_color()
                #self.nine_grid[int(pikapika / 6)][int((pikapika % 6) / 2)].configure(bg="red")
                self.set_nine_grid_color(int(pikapika / 6), int((pikapika % 6) / 2))
            pikapika = now % 18
            print(pikapika, '/ 18')
    
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

