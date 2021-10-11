import cv2
import os

# 加载训练数据文件
recognizer = cv2.face.LBPHFaceRecognizer_create()
#　加载训练数据
recognizer.read('train/zhao_trainer.yml')
# 名称
names = []
#　警报全局变量
warningtime = 0

# md5加密模块
def Md5(str):
    pass

# 短信反馈

# 警报模块
def Warning():
    pass

#　准备识别的图片
def Face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(r'D:\SOFT\Anaconda3_202105\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    # face_detector = cv2.CascadeClassifier(r'D:\SOFT\Anaconda3_202105\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_defaultq.xml')
    face = face_detector.detectMultiScale(gray, 1.08, 5, cv2.CASCADE_SCALE_IMAGE, (100,100), (900,900))
    # face = face_detector.detectMultiScale(gray)

    for x,y,w,h, in face:
        #　人脸矩形框识别
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # 人脸圆形框识别
        cv2.circle(img,center=(x+w//2, y+h//2),radius=w//2, color=(0,0,255), thickness=2)
        # 人脸识别 confidence为 置信评分
        ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence > 88:
            #　global全局作用域关键字
            # global warningtime
            # warningtime += 1
            # if warningtime > 100:
            #     Warning()
            #     warningtime = 0
            # 置信数大于88，人脸数据不匹配，显示未知
            cv2.putText(img, 'unknow', (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
        else:
            # 　置信系数小于88，人脸数据匹配，显示姓名
            # cv2.putText(img, str(names[ids-1]), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.putText(img, 'zhao', (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        print('confidence ',confidence)
    cv2.imshow('result', img)

# 姓名标签
def name():
    path = 'train_img/'
    # names = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.',2)[1])
        names.append(name)

# 导入人脸检测的视频
cap = cv2.VideoCapture('1.mp4')
# cap = cv2.VideoCapture('2.mp4')
# cap = cv2.VideoCapture('3.mp4')
name()
while True:
    flag,frame = cap.read()
    if not flag:
        break
    Face_detect_demo(frame)
    if ord('q') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
