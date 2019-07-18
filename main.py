import tkinter as tk
from tkinter import Frame, Label, Button, YES, LEFT, BOTH, X, DISABLED, NORMAL, RIGHT, TOP, BOTTOM
from tkinter.messagebox import showinfo, askyesno
from tkinter.simpledialog import askstring
from PIL import ImageTk, Image
import cv2
import numpy as np
import threading
import time
import os

import dlib_utils
import utils


from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing import image


camera_width, camera_height = 640, 480
img_width, img_height = 128, 32
model_path = r'D:\DL\code\weight\20190608\two_eyes_gaze_keras_resnet50_model.json'
model_weight_path = r'D:\DL\code\weight\20190608\two_eyes_gaze_keras_resnet50_model.h5'


mode_list = ['Demo for big grid', 'Demo for big camera', 'Collect data']
grid_frames = [[3,300],[4,300],[5,300]]

mode = mode_list[0]

for_exe = False

name = 'jie4'
gf = 0

grid_size = grid_frames[gf][0]
frames_of_grid = grid_frames[gf][1]


grid_row = grid_size
grid_col = grid_size
grid_button_width = 200
grid_button_height = 200
grid_type = '{0:d}x{1:d}'.format(grid_row, grid_col)

gaze_frames = 8

change_frequency_frames = 50

r_col = [3,3,3]
grid_row = len(r_col)
grid_col = 3
r_total_col = utils.sequence_sum(r_col)
grid_type = '{0:d}x{1:d}'.format(grid_row, grid_col)
#grid_type = '{0:d}@{1:d}'.format(grid_row, sum(r_col))




class Application(Frame):
    def __init__(self, master):
        # init main window 
        master.title("Gaze Block Estimation")
        master.resizable(False, False)


        self.load_model_thread()
        #self.model = None

        #self.frame1 = Frame(master)
        self.frame2 = Frame(master)
        self.grid_image = utils.cvimg2tkimg(cv2.imread('pixel.png'))

        
        self.telephone_image = utils.cvimg2tkimg(cv2.imread('call.png'))
        self.message_image = utils.cvimg2tkimg(cv2.imread('message.png'))
        self.voice_image = utils.cvimg2tkimg(cv2.imread('music.png'))
        self.exit_image = utils.cvimg2tkimg(cv2.imread('exit.jfif'))
        self.frame_grids = [0 for x in range(grid_row)]
        for i in range(grid_row):
            self.frame_grids[i] = Frame(master)
        # init left grid (for gaze blocks)
        #self.grids = [[0 for x in range(grid_row)] for x in range(grid_col)] 
        self.grids = utils.create_grids(r_col)

        for row in range(grid_row):
            for col in range(len(self.grids[row])):
                self.grids[row][col] = Button(self.frame_grids[row], 
                                                image=self.grid_image,
                                                width=grid_button_width, 
                                                height=grid_button_height,
                                                bg="white"
                                                )
                self.grids[row][col].grid(row=0, column=col)
        self.grids[0][0].config(image=self.telephone_image)
        self.grids[0][1].config(image=self.message_image)
        self.grids[0][2].config(image=self.voice_image )
        self.grids[2][2].config(image=self.exit_image )                    
        self.grids[2][0].bind("<Button-1>", self.call_auto_close)


        self.thread_o = threading.Thread(target=self.other_camera_thread)
        self.thread_w_signal = False
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)    
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)#幀率

        # init model thread
        #self.model = None
        #self.thread_m = threading.Thread(target=self.load_model_thread)

        # init frame counter
        self.frame_counter = np.full((3), -1)

        self.set_window_size(mode)




        
    def set_window_size(self, select_mode):
        root.attributes('-fullscreen', True)
        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
        if select_mode == mode_list[0]:
            for row in range(grid_row):
                self.frame_grids[row].pack(side=TOP)
                for col in range(len(self.grids[row])):
                    #self.grids[row][col].config(height=root.winfo_screenheight()//grid_row, width=(root.winfo_screenwidth()*3/4)//len(self.grids[row]))
                    self.grids[row][col].config(height=root.winfo_screenheight()//grid_row, width=root.winfo_screenwidth()//len(self.grids[row]))
            self.thread_w_signal = True
            self.thread_o.start()


        elif select_mode == mode_list[1]:
            print('nothing')
        elif select_mode == mode_list[2]:
            #self.frame1.pack(fill=X)
            for row in range(grid_row):
                self.frame_grids[row].pack(fill=X)
                for col in range(len(self.grids[row])):
                    self.grids[row][col].config(height=root.winfo_screenheight()//grid_row, width=root.winfo_screenwidth()//len(self.grids[row]))
                    self.grids[row][col].bind("<Button-1>", lambda event, r=row, c=col: self.click_grids_button(r, c))
                    self.grids[row][col].bind("<Button-3>", self.on_closing_evt)


    def popup_hello(self):  # for command
        showinfo("Hello", ">_<")




    def click_grids_button(self, r, c):
        
        # init show a color
        self.set_grid_color(r, c, '#008844')
        frame_count = 0

        # init dir
        save_path = r'\eyes\{0:s}\{1:s}'.format(name, grid_type)
        utils.make_dir(save_path)
        video_path = os.path.join(save_path, '{0:s}_{1:s}_{2:d}.avi'.format(name, grid_type, r_total_col[r]+c))
        

        print(video_path)

        if not os.path.isfile(video_path):
            # init time and writer
            start = time.time()
            out = cv2.VideoWriter(video_path, self.fourcc, self.fps, (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while(frame_count < frames_of_grid):
                ret, frame = self.capture.read() 
                
                if ret == True:
                    out.write(frame)
                    frame_count += 1
                    if frame_count % change_frequency_frames == 0:
                        #self.set_grid_color(r, c, utils.random_color())
                         self.set_grid_color(r, c, utils.random_color2())
                    root.update()

            out.release()
            end = time.time() - start
            print(end, 'secs')

            # disable button
            #self.grids[r][c].unbind("<Button-1>")

        else:
            showinfo("Warning", "Video has existed!")

        self.reset_grid_color()
        root.update()
        self.grids[r][c].configure(state=DISABLED)


    def set_grid_color(self, row, col, color="red"):
        self.grids[row][col].config(bg=color)

    def reset_grid_color(self, color="black"):
        for row in range(grid_row):
            for col in range(len(self.grids[row])):
                self.grids[row][col].configure(bg=color)

    def popup_qq_event(self):
        showinfo("QQ", "QAQ")

    def call_auto_close(self, event):
        showinfo("提示", "即將撥打電話給朋友...")
        '''
        self.pop_window = tk.Tk()
        self.pop_window.title('Hello')
        self.pop_window['width'] = 400
        self.pop_window['height'] = 300
        self.pop_window_text = tk.Text(self.pop_window, width=300)
        self.pop_window_text.insert('0.0', 'Today is good.')
        self.pop_window_text.place(x=10, y=10, width= 380, height=230)
        #self.pop_window.mainloop()
        ac = threading.Thread(target=self.auto_close())
        ac.start()
        '''

    def auto_close(self):
        '''
        for i in range(3):
            time.sleep(1)
            print('time!!!!!!!!!!!!')
        #self.pop_window.destroy()
        
        print('QAQAQAQAQAQAQAQAQAQAQQAQAQAQAQ')
        self.pop_window.iconify()
        '''
    

    def open_camera(self, event):
        self.thread_w_signal = True
        self.thread_w.start()

    def load_model(self, event):
        self.thread_m.start()

    def load_model_thread(self):
        print('loading model...')
        with open(model_path, 'r') as file:
            model_json = file.read()
            self.model = model_from_json(model_json)
        self.model.load_weights(model_weight_path)
        self.model.predict(np.zeros((1, img_height, img_width, 3)))  # before using model, must predict once to avoid 'Tensor is not an element of this graph.'

        print('loading completed!')


    def other_camera_thread(self):
        print('start webcam_thread')
        # 設定影像的尺寸大小
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        roi = [0,0,0,0]
        # 刷新縮圖
        while(self.thread_w_signal):
            start = time.time()
            ret, frame = self.capture.read() 
            catch, tkframe, two_eyes, roi = dlib_utils.find_face_and_crop_two_eyes(frame, roi, img_width, img_height)

            cv2.imshow('Camera window', tkframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
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
                        self.frame_counter[2] = 0
                        self.reset_grid_color("white")
                    
                    # change color
                    if self.frame_counter[2] >= (gaze_frames-1):
                        #self.reset_grid_color("white")
                        to_change_row, to_change_col = utils.get_row_col(r_col, top_class) ## solution?
                        self.set_grid_color(to_change_row, to_change_col)                  ## solution?
                        self.frame_counter[2] -= gaze_frames
                        
                        if self.frame_counter[1] == 0:
                            showinfo("提示", "即將撥打電話給朋友...")
                        if self.frame_counter[1] == 1:
                            showinfo("提示", "發送訊息給朋友...")
                        if self.frame_counter[1] == 2:
                            showinfo("提示", "播放音樂...")
                        if self.frame_counter[1] == 8:
                            showinfo("提示", "關閉程式")
                            self.thread_w_signal = False
                            self.capture.release()
                            cv2.destroyAllWindows()
                            root.destroy()
                    
                    print(self.frame_counter)
                    self.frame_counter[0] = self.frame_counter[1]
                else:
                    roi = [0,0,0,0]


            end = time.time()
            seconds = end - start
            print('fps', (1/seconds))
        self.capture.release()
        cv2.destroyAllWindows()

    def on_closing(self):
        ans = askyesno(title='Quit', message='Do you want to quit?')
        if ans:
            self.thread_w_signal = False
            self.capture.release()
            root.destroy()
        else:
            return
    def on_closing_evt(self, event):
        ans = askyesno(title='Quit', message='Do you want to quit?')
        if ans:
            self.thread_w_signal = False
            self.capture.release()
            root.destroy()
        else:
            return




root = tk.Tk()
app = Application(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
