# 在一张图片上检测圆
import cv2
import os
import numpy as np
from itertools import chain


def show_img(img):
     #可调整窗口显示图像，鼠标点进图像，键盘回车，退出程序
    cv2.namedWindow('contour', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('contour', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_circle(img):
    """在一张图片上检测圆
    img: 必须是二值化的图
    """
    # img = img * 255
    #show_img(img)
    img_bgr = np.stack([img,img,img],axis=-1)
    #show_img(img)
    # param2越小，检测到的圆越多
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,
                                                            1,#图像解析的反向比例。1为原始大小，2为原始大小的一半
                                                            100,#圆心之间的最小距离。过小会增加圆的误判，过大会丢失存在的圆
                                                            param1=25,#Canny检测器的高阈值
                                                            param2=20,#检测阶段圆心的累加器阈值。越小的话，会增加不存在的圆；越大的话，则检测到的圆就更加接近完美的圆形
                                                            minRadius=200,#检测的最小圆的半径
                                                            maxRadius=999999)#检测的最大圆的半径
    #print(np.array(circles))

    circles = np.uint16(np.around(circles))
    print(circles.shape)
    # 因为我这里只检测一个圆，需要检测多个圆的话，就遍历circles
    #for i in circles[0, :]:
    #    cv2.circle(img_bgr, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
    #    cv2.circle(img_bgr, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    #show_img(img_bgr)
    assert len(circles)==1,f'{circles},not qualify only a circle!'
    (x,y,r) = circles[0][0]
    # input(circles[0,:])
    show = False
    if show==False:
        cv2.circle(img_bgr,(x,y),r,(0,0,255),2)
        # 圆心
        cv2.circle(img_bgr,(x,y),2,(0,255,0),3)
        cv2.imshow('w',img_bgr)
        cv2.waitKey(0)
    # 这里的x对应w,y对应d
    return (x,y,r),img_bgr # 返回横纵坐标和半径、rgb图

#图像处理
def tuxiangchuli(im):
    #缩小图像
    im = im[cut:(im.shape[0] - cut-200),cut:(im.shape[1]-cut)]
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    # 转换为灰度图片
    imgray0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #高斯滤波模糊平滑图像
    imgray1 = cv2.GaussianBlur(imgray0,(5,5),0)
    #自适应二值化,工件为黑色
    imgray2=cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,9) # 自适应二值化
    #高斯滤波模糊平滑图像
    imgray3 = cv2.GaussianBlur(imgray2,(3,3),0)
    #开运算，去白色毛刺和噪点
    kernel = np.ones((3,3),np.uint8)
    imgray4 = cv2.morphologyEx(imgray3, cv2.MORPH_OPEN, kernel, 1)
    #闭运算，连贯线条
    imgray4 = cv2.morphologyEx(imgray4, cv2.MORPH_CLOSE, kernel, 1) 
    #自适应二值化，消去白色噪声
    ret3,imgray5 = cv2.threshold(imgray4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # STSU 再次二值化
    #卷积运算
    #模糊处理核，减弱像素点和周围的分别
    kernel = np.array((
        [0.0625,0.125,0.065],
        [0.123,0.25,0.125],
        [0.0625,0.125,0.0625]
    ),dtype="float32")
    #kernel = np.ones((5,5),np.float32)/25 #师兄的
    filter2D = cv2.filter2D(imgray5,-1,kernel)
    filter2D = cv2.medianBlur(filter2D,5)#中值滤波
    #二值化，清晰边界
    ret3,imgray6 = cv2.threshold(filter2D,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 再次二值化
    #去除小的黑色块噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #定义矩形结构元素 内核[0 1 0,1 1 1,0 1 0]
    opened1 = cv2.morphologyEx(imgray6, cv2.MORPH_CLOSE, kernel,iterations=5)
    #高斯滤波模糊平滑图像,弱化运算产生棱角
    opened1 = cv2.GaussianBlur(opened1,(5,5),0)
    #二值化，清晰边界
    _,opened1 = cv2.threshold(opened1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im,imgray5

#找轮廓
def biankuang(imgray3):
    grad_x = cv2.Sobel(imgray3, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(imgray3, cv2.CV_16S, 0, 1)
    canny_0 = 20
    imgray3 = cv2.Canny(grad_x, grad_y, canny_0,canny_0 * 3 )
    return imgray3


if __name__ == '__main__':
    file_path = "yuan.jpeg"
    # 读取彩色图片
    im = cv2.imread(file_path)
    # print(np.unique(img))
    #缩小图像
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    cut = 30#裁剪边框尺寸,用来去除边缘的噪声03-1
    medianBlur_size = 15 #均值滤波核大小，单数1- 9,去掉中值滤波
    canny_0 =30#边缘检测低阀值0-8,03-14,2-20,1-25
    threshVal = 50  #根据图像情况设置阀值1-50,0-120,不过好像加上canny后就没用了
    clean_data = 400#数据清洗去除异常值阀值1-6.1 2-6.55
    #图像处理
    im, image_erzhi = tuxiangchuli(im)
    #找轮廓
    image_biankuang = biankuang(image_erzhi)
    '''#找轮廓
    #_, contours, hierarchy = cv2.findContours(image_biankuang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#与python版本有关
    contours, hierarchy = cv2.findContours(image_biankuang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #list转换为数组(规整)
    contours_array = np.array(contours,dtype = object)
    #数组转换为list（降唯）
    contours_list_1 = list(chain(*contours_array))
    #轮廓排序
    contours_list_1.sort(key=lambda c: cv2.contourArea(c), reverse=True)
     #转为数组（方便操作数据）
    contours_array_1 = np.array(contours_list_1)#(2797,1,2)
    cnt = contours_list_1[0]
    #print(hierarchy)
    '''
    (x,y,r),_ = detect_circle(image_biankuang)
    cv2.circle(im,(x,y),r,(0,0,255),2)
    # 圆心
    cv2.circle(im,(x,y),2,(0,255,0),3)
    cv2.putText(im,  '(' + str(round(x))+',' + str(round(y)) + '),r:' + str(round(r)), (0,y) , cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    show_img(im)
   
   


 





