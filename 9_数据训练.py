import os
import cv2
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    # 储存人脸数据
    facesSamples = []
    # 储存姓名数据
    ids = []
    # 储存图片信息
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv2.CascadeClassifier(r'D:\SOFT\Anaconda3_202105\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片，灰度化PIL有9种不同模式
        PIL_image = Image.open(imagePath).convert('L')
        # 将图片转化为数组，黑白深浅
        img_numpy = np.array(PIL_image, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        #　获取图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容照片
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h, x:x+w])
    # 打印面部特征
    print('id : ', id)
    print('fs : ', facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = 'train_img/'
    # 获取图像数组和Id标签，姓名
    faces, ids = getImageAndLabels(path)
    # 加载识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))

    # 保存文件
    recognizer.write('train/zhao_trainer.yml')
    print('模型已保存！')