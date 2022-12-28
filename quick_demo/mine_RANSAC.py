import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
import os
from itertools import chain
from numpy.linalg import det

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
def get_circle(p1,p2,p3):
    '''
    ###########################################################################
    # 由圆上三点确定圆心和半径
    ###########################################################################
    # INPUT
    # p1   :  - 第一个点坐标, list或者array 1x3
    # p2   :  - 第二个点坐标, list或者array 1x3
    # p3   :  - 第三个点坐标, list或者array 1x3
    # 若输入1x2的行向量, 末位自动补0, 变为1x3的行向量
    ###########################################################################
    # OUTPUT:x,y,r   x,y,z,r
    # pc   :  - 圆心坐标, array 1x3
    # r    :  - 半径, 标量
    ###########################################################################
    # 调用示例1 - 平面上三个点
    # pc1, r1 = points2circle([1, 2], [-2, 1], [0, -3])
    # 调用示例2 - 空间中三个点
    # pc2, r2 = points2circle([1, 2, -1], [-2, 1, 2], [0, -3, -3])
    ###########################################################################
    参考：
        https://blog.csdn.net/Sppy_z/article/details/104877864
        https://copyfuture.com/blogs-details/202212050540427299
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)
    
    # 输入维度检查
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None
   
    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
  
    #计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b。
    # 如果a,b是向量数组，则向量在最后一维定义。该维度可以为2，也可以为3. 为2的时候会自动将第三个分量视作0补充进去计算。
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02) # @装饰器的格式来写的目的就是为了书写简单方便
    # temp03 @ temp03中的@ 含义是数组中每个元素的平方之和
    if temp < 10**-6:
        print('\t三点共线, 无法确定圆')
        return None

    temp1 = np.vstack((p1, p2, p3)) # 行拼接
    temp2 = np.ones(3).reshape(3, 1) # 以a行b列的数组形式显示
    mat1 = np.hstack((temp1, temp2)) # size = 3x4
    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1)) # axis=1相对于把每一行当做列来排列
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)
    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    # 使用 stack，可以将一个列表转换为一个numpy数组，当axis=0的时候，和 使用 np.array() 没有什么区别，
    # 但是当 axis=1的时候，那么就是对每一行进行在列方向上进行运算，也就是列方向结合，
    # 此时矩阵的维度也从（2,3）变成了（3,2）
    # hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5)) # size = 4x5
    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])
    pc = -np.array([B, C, D]) / 2 / A
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)
    if num1 == 2:
        return pc[0], pc[1], r
    else:
        return pc[0], pc[1], pc[2], r



def get_circle2(p1, p2, p3):
    x21 = p2.x - p1.x
    y21 = p2.y - p1.y
    x32 = p3.x - p2.x
    y32 = p3.y - p2.y
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = p2.x * p2.x - p1.x * p1.x + p2.y * p2.y - p1.y * p1.y
    xy32 = p3.x * p3.x - p2.x * p2.x + p3.y * p3.y - p2.y * p2.y
    print(x21,xy21,xy32)
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
    x0 = (xy21 - 2 * y0 * y21) / (2 * x21)
    R = ((p1.x - x0) ** 2 + (p1.y - y0) ** 2) ** 0.5
    return x0, y0, R
 
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

def dec_points(img, p = 255):
    '''
    检测图片中像素值为p的点
    输入：
        img:二维的二值化图像
        p:int 要检测的像素值
    输出：
        列表[ [x1, y1], [x2, y2], [x3, y3] ]
    '''
    img = np.array(img)
    points = np.where(img == p)
    #print(points[0].size)
    points = [ [points[0][i], points[1][i] ] for i in range(points[0].size) ] 
    #print(points)
    return points

def random_points(img, size = 3):
    '''
    随机选取图片中的size个点
    参数：
        img:二维的二值化图像
        size:默认采样3个点
    返回：
        列表[ [x1, y1], [x2, y2], [x3, y3] ]
    '''
    #img size
    size_x = img.shape[0]
    size_y = img.shape[1]

    #根据图片尺寸生成随机点索引列表
    x_sample_index = random.sample(range(size_x),3)
    y_sample_index = random.sample(range(size_x),3)
    
    points = []
    for i in range(size):
        points.append( [x_sample_index[i], y_sample_index[i]] )
    
    return points

def dec_random_points(img, num = 3, p = 255):
    '''
    随机选取图片中指定像素值的size个点
    参数：
        img:二维的二值化图像
        num:默认采样3个点
        p:int 要检测的像素值
    返回：
        列表[ [x1, y1], [x2, y2], [x3, y3] ]
    '''
    #img size
    img_points = dec_points(img)
    size = len(img_points)
    #print(size)

    #根据图片尺寸生成随机点索引列表
    sample_index = random.sample(range(size),num)
    #print(sample_index)
    
    points = []
    for i in range(num):
        points.append( img_points[sample_index[i]] )
    #print(points)
    return points

def get_RANSAC_iters(t, P = 0.99,  n = 3, plus = 2):
    '''
    估算迭代次数
    输入：
        t:(0,1),满足方程的点占全部点的比例
        P:模型正确的概率(选取到的点都是内点的概率),默认0.99
        n:计算模型选取的点数,默认为圆3
        plus:由于随机点共线的情况，需要额外增加迭代次数,默认扩大plus倍,默认为2
    输出:
        迭代次数:int
    '''
    iters = math.log(1 - P) / (math.log(1 - t ** n))
    iters = int(iters)
    iters = iters * plus
    return iters

def RANSAC_circle(img, t = 0.5, P=255, r_min = 100):
    '''
    RANSAC方法拟合圆
    参数：
        img:二维的二值化图像
        t:(0,1),满足方程的点占全部点的比例
        P:int 要检测的像素值
        r_min:所拟合圆的半径不小于该值,默认
    返回：
        圆参数(x,y,r)
    '''
    #提取255像素值的点
    points = dec_points(img, P)
    size = len(points)

    #用于可视化显示，在其上画彩色圆等
    img_bgr = np.stack([img,img,img],axis=-1)
    
    # 使用RANSAC算法估算模型
    # 数据和模型之间可接受的差值
    sigma = 0.1
    # 最好模型的参数估计和内点数目
    best_x0 = 0
    best_y0 = 0
    best_r = 5
    pretotal = 0
    # 希望的得到正确模型的概率
    p = 0.99
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    #参考程序手动设置100000
    iters = get_RANSAC_iters(t, p)#0.05,0.99,i = 36844,0.01,0.7,i = 1203977
    print(iters)

    for i in range(3000):
        if i % 1000 == 0:
            print('已经迭代', i ,'次')
        # 随机选取3个点
        p1, p2, p3 = dec_random_points(img)
        
        #3点求圆方程
        #可以替换不同方案计算圆和对应内点
        #p1, p2, p3 = Point(x_1, y_1), Point(x_2, y_2), Point(x_3, y_3)
        try:
            (x0, y0, r) = get_circle(p1, p2, p3)
            #print(x0,y0,r)
        except:
            continue
            
        if r > r_min:
            # 算出内点数目
            total_inlier = 0
            for index in range(size):
                r_estimate2 = (points[index][0] - x0) ** 2 +  (points[index][1] - y0) **2
                if abs(r_estimate2 - r ** 2) < sigma:
                    total_inlier = total_inlier + 1

            # 判断当前的模型是否比之前估算的模型好
            if total_inlier > pretotal:
                iters = math.log(1 -p) / math.log(1 - pow(total_inlier / (size), 2))
                pretotal = total_inlier
                best_x0 = x0
                best_y0 = y0
                best_r = r

            # 判断是否当前模型已经符合超过一半的点
            if total_inlier > size * t:
                break
        else:
            continue
    print(best_x0,best_y0,best_r)
    return (best_x0, best_y0, best_r)

#图像处理
def tuxiangchuli(im):
    #缩小图像
    im = im[cut:(im.shape[0] - cut),cut:(im.shape[1]-cut)]
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    # 转换为灰度图片
    imgray0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #高斯滤波模糊平滑图像
    imgray1 = cv2.GaussianBlur(imgray0,(5,5),0)
    #自适应二值化,工件为黑色，图片，上限，自适应二值算法ADAPTIVE_THRESH_MEAN_C，提取亮区域THRESH_BINARY（暗区域THRESH_BINARY_INV），像素邻域大小（单数），偏移值调整量（单数）
    imgray2=cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,9) # 自适应二值化11,9
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
    file_path = '1.jpeg'
    #file_path = 'shaoyuan.png'
    #file_path = "yuan.jpeg"
    # 读取彩色图片
    im = cv2.imread(file_path)
    # print(np.unique(img))
    #缩小图像
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    cut = 0#裁剪边框尺寸,用来去除边缘的噪声03-1
    medianBlur_size = 15 #均值滤波核大小，单数1- 9,去掉中值滤波
    canny_0 =30#边缘检测低阀值0-8,03-14,2-20,1-25
    threshVal = 50  #根据图像情况设置阀值1-50,0-120,不过好像加上canny后就没用了
    clean_data = 400#数据清洗去除异常值阀值1-6.1 2-6.55
    #图像处理
    im, image_erzhi = tuxiangchuli(im)
    #points = dec_points(image_erzhi)
    #show_img(image_erzhi)
    #找轮廓
    image_biankuang = biankuang(image_erzhi)
    show_img(image_erzhi)
    show_img(image_biankuang)
    #show_img(image_biankuang)

    #points = dec_points(image_biankuang)
    
    #(x,y,r),_ = detect_circle(image_biankuang)
    
    (x, y, r) = RANSAC_circle(image_biankuang)
    x_int = int(x)
    y_int = int(y)
    image_biankuang = np.stack([image_biankuang,image_biankuang,image_biankuang],axis=-1)
    cv2.circle(image_biankuang,(x_int, y_int),int(r),(0,0,255),2)
    cv2.circle(im,(x_int, y_int),int(r),(0,0,255),2)
    # 圆心
    cv2.circle(image_biankuang,(x_int, y_int),2,(0,255,0),3)
    cv2.circle(im,(x_int, y_int),2,(0,255,0),3)
    cv2.putText(im,  '(' + str(round(x))+',' + str(round(y)) + '),r:' + str(round(r)), (0,y_int) , cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    save_name_title = file_path.split('.')[0]
    cv2.imwrite(save_name_title + '_biankuang.jpg', image_biankuang)
    cv2.imwrite(save_name_title + '_im.jpg',im)
    show_img(image_biankuang)
   