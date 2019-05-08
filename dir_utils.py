import os 


def make_dir(path):
    if not os.path.isdir(path):
        last_dir = os.path.dirname(path)
        make_dir(last_dir)
        os.mkdir(path)
