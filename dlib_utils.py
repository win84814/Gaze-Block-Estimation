from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import time
import os
import utils
predictor_path = 'shape_predictor_68_face_landmarks.dat'
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
    #img = imutils.resize(img, width=512)  # 調整圖片寬度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        tx, ty, tw, th = get_rectangle(shape[36:47, :])
    if tx == 0 or ty == 0:
        return img
    cv2.rectangle(img,
                    (tx-10, ty-10),
                    (tx+tw+10, ty+th+10),
                    (255, 0, 0),
                    1)
    print('size:', (tw+20), (th+20))
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


def find_face_and_crop_two_eyes(img, roi, resize_width=128, resize_height=32):
    img_thumbnail = img

    if roi[2] != 0 and roi[3] != 0 :
        new_img = img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        tx, ty, tw, th = get_rectangle(shape[36:47, :])
        #break  # detect only one person
    

        tx += roi[0]
        ty += roi[1]
        roi = [x+roi[0], y+roi[1], w, h]
        roi = extension_roi(img, roi)
        #x += roi[0]
        #y += roi[1]

    if tw == 0 or th == 0:
        return False, img, img, roi
    else:
        new_img = img[max(ty-10, 0):ty+th+10, max(tx-10, 0):tx+tw+10]
        new_img = cv2.resize(new_img, (resize_width, resize_height))
        
        #cv2.rectangle(img_thumbnail, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_thumbnail,
                      (tx-10, ty-10),
                      (tx+tw+10, ty+th+10),
                      (0, 255, 0),
                      1)
        return True, img_thumbnail, new_img, roi
    
def find_one_two_eyes(img):
    #img_thumbnail = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    tx, ty, tw, th = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        if w > 50 and h > 50:
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
    roi = [0,0,0,0]
    while(cap.isOpened()):
        ret, frame = cap.read()
        start = time.time()
        catch, frame, two, roi = find_face_and_crop_two_eyes(frame, roi) 
        if not catch:
            roi = [0,0,0,0]
        
        #cv2.imwrite(r'D:\DL\code\Gaze-Block-Estimation\my_eyes.png', frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        end = time.time()
        seconds = end - start
        print('fps', (1/seconds))
    cap.release()
    cv2.destroyAllWindows()

def time_of_crop():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start, end = 0, 0
    #frame_count = 0
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

    utils.make_dir(save_path)
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
    save_path = r'D:\eyes\dennisliu\4@14'
    video_path = os.path.join(save_path, 'dennisliu_4@14_10.avi')
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = find_face_and_eyes(frame)
            frame_count += 1
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('total frames', frame_count)


def get_eyes_data(path):
    print(path)
    videos_path = utils.get_files(path, '*.avi')
    print(videos_path)

    all_valid_frames = 0
    all_frames = 0
    for i in range(len(videos_path)):
        print(videos_path[i])
        save_path = os.path.join(os.path.dirname(videos_path[i]), utils.get_last_number(videos_path[i]))
        r_path = os.path.join(save_path, 'r')
        l_path = os.path.join(save_path, 'l')
        t_path = os.path.join(save_path, 't')
        utils.make_dir(save_path)
        utils.make_dir(r_path)
        utils.make_dir(l_path)
        utils.make_dir(t_path)

        cap = cv2.VideoCapture(videos_path[i])

        frame_count = 0
        valid_count = 0
        # 以迴圈從影片檔案讀取影格，並顯示出來
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:        
                catch, one_r, one_l, two = find_one_two_eyes(frame)
                if catch:
                    cv2.imwrite(os.path.join(r_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(videos_path[i]), frame_count)), one_r)
                    cv2.imwrite(os.path.join(l_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(videos_path[i]), frame_count)), one_l)
                    cv2.imwrite(os.path.join(t_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(videos_path[i]), frame_count)), two)
                    valid_count +=1
                frame_count += 1
            else:
                break
        cap.release()
        all_valid_frames += valid_count
        all_frames += frame_count
        print('valid frames', valid_count)
        print('total frames', frame_count)
    
    print('all valid frames', all_valid_frames)
    print('all frames', all_frames)

def get_eyes_data_t(path):
    print(path)
    videos_path = utils.get_files(path, '*.avi')
    print(videos_path)

    all_valid_frames = 0
    all_frames = 0
    for i in range(len(videos_path)):
        print(videos_path[i])
        save_path = os.path.join(os.path.dirname(videos_path[i]), utils.get_last_number(videos_path[i]))
        utils.make_dir(save_path)

        cap = cv2.VideoCapture(videos_path[i])

        frame_count = 0
        valid_count = 0
        # 以迴圈從影片檔案讀取影格，並顯示出來
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:        
                catch, one_r, one_l, two = find_one_two_eyes(frame)
                if catch:
                    cv2.imwrite(os.path.join(save_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(videos_path[i]), frame_count)), two)
                    valid_count +=1
                frame_count += 1
            else:
                break
        cap.release()
        all_valid_frames += valid_count
        all_frames += frame_count
        print('valid frames', valid_count)
        print('total frames', frame_count)
    
    print('all valid frames', all_valid_frames)
    print('all frames', all_frames)

def get_eyes_data_for_one(path):
    save_path = os.path.join(os.path.dirname(path), utils.get_last_number(path))
    r_path = os.path.join(save_path, 'r')
    l_path = os.path.join(save_path, 'l')
    t_path = os.path.join(save_path, 't')
    utils.make_dir(save_path)
    utils.make_dir(r_path)
    utils.make_dir(l_path)
    utils.make_dir(t_path)

    cap = cv2.VideoCapture(path)

    frame_count = 0
    valid_count = 0
    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:        
            catch, one_r, one_l, two = find_one_two_eyes(frame)
            if catch:
                cv2.imwrite(os.path.join(r_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(path), frame_count)), one_r)
                cv2.imwrite(os.path.join(l_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(path), frame_count)), one_l)
                cv2.imwrite(os.path.join(t_path, '{0:s}_{1:3d}.png'.format(utils.get_filename_without_extension(path), frame_count)), two)
                valid_count +=1
            frame_count += 1
        else:
            break
    cap.release()
    print('valid frames', valid_count)
    print('total frames', frame_count)

def extension_roi(img,roi,x=1.5):
    # crop img with roi, then extend x*100% roi.h and roi.w
    max_h = img.shape[0] # height
    max_w = img.shape[1] # width
    extend_min_x = int(roi[0]-roi[2]*((x-1)/2))
    if extend_min_x < 0 : 
        extend_min_x = 0
    extend_min_y = int(roi[1]-roi[3]*((x-1)/2))
    if extend_min_y < 0 : 
        extend_min_y = 0
    extend_max_x = int(roi[0]+roi[2]*(1+(x-1)/2))
    if extend_max_x >= max_w : 
        extend_max_x = max_w - 1
    extend_max_y = int(roi[1]+roi[3]*(1+(x-1)/2))
    if extend_max_y >= max_h : 
        extend_max_y = max_h - 1
    return [extend_min_x, extend_min_y, extend_max_x - extend_min_x, extend_max_y - extend_min_y]

if __name__ == '__main__':
    main()
    #time_of_crop()
    #test_write_video()
    #test_load_video()
    #get_eyes_data_for_one(r'D:\DL\dataset\eyes\wei\5x5\wei_5x5_4.avi')
    # 
    '''
    name = 'ching'
    grid_size = 4

    grid_type = '{0:d}x{1:d}'.format(grid_size, grid_size)
    folder_path = r'D:\DL\dataset\eyes\{0:s}\{1:s}'.format(name, grid_type)

    start = time.time()
    get_eyes_data(folder_path)
    end = time.time() - start
    print(end, 'secs')
    '''
    '''
    name = 'jie4'
    for i in range(3,6):
        grid_size = i
        grid_type = '{0:d}x{1:d}'.format(grid_size, grid_size)
        #folder_path = r'D:\DL\dataset\eyes\{0:s}\{1:s}'.format(name, grid_type)
        folder_path = r'D:\eyes\{0:s}\{1:s}'.format(name, grid_type)
        start = time.time()
        get_eyes_data_t(folder_path)
        end = time.time() - start
        print(end, 'secs')
    '''