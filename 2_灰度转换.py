# 导入CV模块
import cv2 as cv

# 读取图片
img = cv.imread('face1.jpg')

# 显示图像
cv.imshow('read_img', img)

# 灰度转换
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 显示灰度
cv.imshow('gray', gray_img)

# 保存灰度
cv.imwrite('gray_face1.jpg', gray_img)


# 等待
cv.waitKey(0)

# 释放内存
cv.destroyAllWindows()