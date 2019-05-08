from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import time
import os
import dir_utils
predictor_path = 'D:/DL/dataset/dlib/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

name = 'jie'
date = '20190507'

def get_rectangle(points):
    columns = [[row[col] for row in points] for col in range(len(points[1]))]
    
    max_x = max(columns[0])
    min_x = min(columns[0])
    max_y = max(columns[1])
    min_y = min(columns[1])
    return (min_x, min_y, max_x - min_x, max_y - min_y)
    

def find_face_and_eyes(img):
    img = imutils.resize(img, width=512)  # 調整圖片寬度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for i, (x, y) in enumerate(shape):
            # two eyes
            tx, ty, tw, th = get_rectangle(shape[36:47, :])
            
            cv2.rectangle(img,
                          (tx-10, ty-10),
                          (tx+tw+10, ty+th+10),
                          (255, 0, 0),
                          1)
            
    return img


def crop_two_eyes(img, resize_width=128, resize_height=32):
    img = imutils.resize(img)  # 調整圖片寬度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        # two eyes
        tx, ty, tw, th = get_rectangle(shape[36:47, :])

    if tx == 0 or ty == 0:
        return False, img
    else:
        # return img[tx-10:tx+tw+10, ty-10:ty+th+10]
        new_img = img[max(ty-10, 0):ty+th+10, max(tx-10, 0):tx+tw+10]
        new_img = cv2.resize(new_img, (resize_width, resize_height))
        # cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\my_eyes2.png', new_img)
        return True, new_img  # img[ty-10:ty+th+10, tx-10:tx+tw+10]


def find_face_and_crop_two_eyes(img, resize_width=128, resize_height=32):
    img_thumbnail = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        tx, ty, tw, th = get_rectangle(shape[36:47, :])
        #break  # detect only one person
    
    if tx == 0 or ty == 0:
        return False, img, img
    else:
        new_img = img[max(ty-10, 0):ty+th+10, max(tx-10, 0):tx+tw+10]
        new_img = cv2.resize(new_img, (resize_width, resize_height))
        
        cv2.rectangle(img_thumbnail, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_thumbnail,
                      (tx-10, ty-10),
                      (tx+tw+10, ty+th+10),
                      (255, 0, 0),
                      1)
        print((tw+20))
        return True, img_thumbnail, new_img
    
def find_one_two_eyes(img):
    #img_thumbnail = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        tx, ty, tw, th = get_rectangle(shape[36:47, :])
        rx, ry, rw, rh = get_rectangle(shape[36:41, :])
        lx, ly, lw, lh = get_rectangle(shape[42:47, :])
        #break  # detect only one person
    
    if tx == 0 or ty == 0:
        return False, img, img, img
    else:
        two_img = img[max(ty-10, 0):ty+th+10, max(tx-10, 0):tx+tw+10]
        r_img = img[max(ry-10, 0):ry+rh+10, max(rx-10, 0):rx+rw+10]
        l_img = img[max(ly-10, 0):ly+lh+10, max(lx-10, 0):lx+lw+10]
        '''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img,
                      (tx-10, ty-10),
                      (tx+tw+10, ty+th+10),
                      (255, 0, 0),
                      1)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 2)
        '''
        return True, r_img, l_img, two_img

def main():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        catch, frame = crop_two_eyes(frame) 
        #cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\my_eyes.png', frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def time_of_crop():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start, end = 0, 0
    frame_count = 0
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        #catch, one_r, one_l, two = find_one_two_eyes(frame)
        end = time.time() - start
        print(end, 'secs')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_write_video():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_path = r'D:\DL\dataset\eyes\jie\20190508\0'
    video_path = os.path.join(save_path, 'output.avi')
 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)#幀率
    print('fps',fps)

    dir_utils.make_dir(save_path)
    out = cv2.VideoWriter(video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # 寫入影格
            out.write(frame)
            frame_count += 1
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # 釋放所有資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('total frames', frame_count)

def test_load_video():
    save_path = r'D:\DL\dataset\eyes\jie\20190508\0'
    video_path = os.path.join(save_path, 'output.avi')
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('total frames', frame_count)

if __name__ == '__main__':
    #main()
    #time_of_crop()
    #test_write_video()
    test_load_video()