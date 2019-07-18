import os 
import random
import string
import glob
import cv2
from PIL import ImageTk, Image


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

def random_color2():
    colors = ['#003377',  '#227700', '#A42D00', '#008844', '#008888', '#007799', '#888800']
    total_colors = len(colors)
    
    return colors[random.randint(0,total_colors-1)]

def get_filename_without_extension(path):
    fileName = os.path.basename(path)
    fileName = os.path.splitext(fileName)[0]
    return fileName

def get_files(path, ext):
    files = glob.glob(os.path.join(path, ext))
    #for i in range(len(files)):
    #    files[i] = os.path.basename(files[i])
    return files

def get_last_number(fileName):
    fileName = os.path.basename(fileName)
    fileName = os.path.splitext(fileName)[0]
    return fileName.split('_')[-1]

def create_grids(sequence):
    seq = [[] for i in range(len(sequence))]
    for row in range(len(sequence)):
        for col in range(sequence[row]):
            seq[row].append(0)
    return seq

def sequence_sum(sequence):
    seq_sum = [0 for x in range(len(sequence))]
    for i in range(1, len(sequence)):
        seq_sum[i] = seq_sum[i-1] + sequence[i-1]
    return seq_sum

def get_row_col(sequence, index):
    row = 0
    for i in range(len(sequence)):
        if index >= sequence[i]:
            index -= sequence[i]
            row += 1
        else:
            break
    col = index
    return row, col

def cvimg2tkimg(cvimg):        
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cvimg = Image.fromarray(cvimg)
    tkimg = ImageTk.PhotoImage(cvimg)
    return tkimg


if __name__ == '__main__':
    #ret = get_files(r'D:\DL\dataset\eyes\jie\3x3', '*.avi')
    #print(ret)
    #ret = get_last_number('D:\\DL\\dataset\\eyes\\jie\\3x3\\jie_3x3_5.avi')
    #print(ret)
    print('A')