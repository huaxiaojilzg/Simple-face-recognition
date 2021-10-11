# 导入CV模块
import cv2 as cv

# 读取图片
# img = cv.imread('face2.jpg')

# ===============================分割线===============================
# 人脸检测函数
def face_detect_demo(img):
    # 图片灰度转换
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载opencv分类器
    face_detect = cv.CascadeClassifier(r'D:\SOFT\Anaconda3_202105\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(100,100), maxSize=(300,300) )
    # 检测人脸
    for x, y, w, h in face:
        cv.rectangle(img, pt1=(x,y), pt2=(x+w, y+h), color=(0,255,0), thickness=2)
    cv.imshow('result', img)

# 读取摄像头 或者 本地视频
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('1.mp4')
# 读取
# cap.read()





# ===============================分割线===============================
# 等待
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break

# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()