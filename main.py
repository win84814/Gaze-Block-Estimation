import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

#ttk 是 tkinter 另外一組視窗元件的程式庫， ttk 可替視窗元件設定不同的樣式。
class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.pack()
        
        root.title("Gaze Block Estimation")
        root.resizable(False,False)

        self.set_window_size()

    def add_button(self):
        self.button = ttk.Button(self)
        self.button["text"] = "Click Me!"
        self.button["command"] = self.popup_hello
        self.button.pack()

    def set_window_size(self):
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry('%sx%s+%s+%s' % (screen_width, screen_height, 0, 0) )   #center window on desktop

    def popup_hello(self):
        showinfo("Hello", ">_<")
    
    def add_canvas(self):
        cv = tk.Canvas(root,width=200,height=200,bg = 'white')
        cv.create_rectangle(10,10,20,20)
        cv.create_rectangle(30,30,40,40)
        cv.pack()


root = tk.Tk()
app = Application(root)
#app.add_canvas()
root.mainloop()

