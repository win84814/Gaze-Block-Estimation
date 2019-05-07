from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import time
predictor_path = 'D:/DL/dataset/dlib/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


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
        return True, img_thumbnail, new_img
    

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
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    start, end = 0, 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        start = time.time()
        catch, thumbnail, new_img = find_face_and_crop_two_eyes(frame) 
        end = time.time() - start
        print(end, 'secs')
        #if catch:
        #    cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\9\my_face.png', thumbnail)
        #    cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\9\my_eyes.png', new_img)
        #    print(end, 'secs')
        #    break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main()
    time_of_crop()