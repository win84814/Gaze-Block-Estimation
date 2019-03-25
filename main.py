import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

#ttk 是 tkinter 另外一組視窗元件的程式庫， ttk 可替視窗元件設定不同的樣式。
class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.pack()

        self.button = ttk.Button(self)
        self.button["text"] = "Click Me!"
        self.button["command"] = self.popup_hello
        self.button.pack()
        

    def popup_hello(self):
        showinfo("Hello", "Hello Tk!")

root = tk.Tk()
app = Application(root)
root.mainloop()

