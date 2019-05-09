import os 
import random
import string

def make_dir(path):
    if not os.path.isdir(path):
        last_dir = os.path.dirname(path)
        make_dir(last_dir)
        os.mkdir(path)

def random_color():
    color = '#'
    for i in range(6):
        color += random.choice('0123456789abcdef') 
    return color

    