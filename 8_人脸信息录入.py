import cv2 as cv

# 摄像头
cap = cv.VideoCapture(0)

flag = 1
num = 0
name = 'Dapeiqi'
while(cap.isOpened()): # 检测摄像头是否开启
    ret_flag, vshow = cap.read() # 得到每一帧的图像
    cv.imshow('Capture', vshow) # 显示图像
    k = cv.waitKey(1) # 按键判断
    if k == ord('s'):
        cv.imwrite('face_img/' + str(num) + '_' + name + '.jpg', vshow)
        print('success to save' + str(num) + '.jpg')
        print('==========')
        num += 1
    elif k == ord('q'): # 退出
        break

# 释放摄像头
cap.release()
# 释放内存
cv.destroyAllWindows()

