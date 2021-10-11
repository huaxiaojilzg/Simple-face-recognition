# 导入CV模块
import cv2 as cv

# 读取图片
img = cv.imread('face1.jpg')

# 修改尺寸
resize_img = cv.resize(img, dsize=(200,200))

# 显示原图
cv.imshow('face1', img)

# 显示修改图
cv.imshow('resize_img', resize_img)

# 打印原图片尺寸
print('原图', img.shape)

# 打印修改图片尺寸
print('修改图', resize_img.shape)

# 等待
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()