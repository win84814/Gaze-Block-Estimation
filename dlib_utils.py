from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

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


def crop_two_eyes(img):
    img = imutils.resize(img)  # 調整圖片寬度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for i, (x, y) in enumerate(shape):
            # two eyes
            tx, ty, tw, th = get_rectangle(shape[36:47, :])
            '''
            cv2.rectangle(img, 
                          (tx-10, ty-10),
                          (tx+tw+10, ty+th+10),
                          (255, 0, 0),
                          1)
            '''
    #return img[tx-10:tx+tw+10, ty-10:ty+th+10]
    #print(tx, ty, tw, th)
    if tx == 0 or ty == 0:
        return False, img
    else:
        #return img[tx-10:tx+tw+10, ty-10:ty+th+10]
        new_img = img[max(ty-10,0):ty+th+10, max(tx-10,0):tx+tw+10]
        new_img = cv2.resize(new_img, (128,32))
        return True, new_img #img[ty-10:ty+th+10, tx-10:tx+tw+10]


def main():
    '''
    image_path = 'D:/DL/dataset/columbia_gaze_data_set/Columbia Gaze Data Set/0006/0006_2m_0P_0V_0H.jpg'
    #image_path = r'D:\DL\dataset\MPIIGaze\Data\Original\p00\day13\0203.jpg'
    image = cv2.imread(image_path)#輸入圖片實參則讀入圖片

    # show the face number 人臉序號的標記（可識別多張）
    #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #image = find_face_and_eyes(image)
    image = find_face_and_eyes(image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    '''
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        catch, frame = crop_two_eyes(frame) 
        cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\my_eyes.png',frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()