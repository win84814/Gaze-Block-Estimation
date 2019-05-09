import os 
import random
import string
import glob

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

if __name__ == '__main__':
    ret = get_files(r'D:\DL\dataset\eyes\jie\3x3', '*.avi')
    print(ret)
    #ret = get_last_number('D:\\DL\\dataset\\eyes\\jie\\3x3\\jie_3x3_5.avi')
    #print(ret)