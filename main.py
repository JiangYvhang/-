import cv2
import numpy as np

from drawer import Drawer
from detectors import RansacCircleDetector, HoughCircleDector

par = {
    'load_par':{
        'file_path':'images/shaoyuan.png'#'images/1.jpeg'
    },
    'process_par':{
        'area':0,                                 #提取区域，0:暗,1:亮
        'area_size':11,                    #计算邻域大小
        'bias':9,                                 #偏移量
    },

    'drawer_par':{
        'wait':0,                                  #图像显示等待时长,0:等待按键,5:5ms
    },

    'detection_par':{
        'P':255,                                    #提取的像素值
        'type':'circle',                        #检测类型
        'algorithm':'RANSAC',       #检测算法
        'RANSAC':{                              #概率结果，不一定每次都一样，但很容易通过半径过滤
                't':0.5,                               #待拟合点占轮廓提取后所有P像素值点的比例,(0,1)
                'P':255,                            #想提取的像素值
                'r_min':160,                  #拟合圆的最小半径
                'p':0.99,                          # 希望的得到正确模型的概率
                'sigma':0.005,                   # 数据和模型之间可接受的差值
                'iters':5000,                   #迭代次数
        },
        'Hough':{
                'bili':1,                             #图像解析的反向比例。1为原始大小，2为原始大小的一半
                'dist_min':100,            #圆心之间的最小距离。过小会增加圆的误判，过大会丢失存在的圆
                'can':25,                         #Canny检测器的高阈值
                'acc':20,                          #检测阶段圆心的累加器阈值。越小的话，会增加不存在的圆；越大的话，则检测到的圆就更加接近完美的圆形
                'r_min':200,                  #检测的最小圆的半径
                'r_max':999999,          #检测的最大圆的半径
        }
    },
}

#图像处理
def tuxiangchuli(im):
    #缩小图像
    im = im[cut:(im.shape[0] - cut),cut:(im.shape[1]-cut)]
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    # 转换为灰度图片
    imgray0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #高斯滤波模糊平滑图像
    imgray1 = cv2.GaussianBlur(imgray0,(5,5),0)
    #用计算代替判断，cv2.THRESH_BINARY + _INV * (1-par)
    if par['process_par']['area']:
        #自适应二值化,工件为黑色，图片，上限，自适应二值算法ADAPTIVE_THRESH_MEAN_C，提取亮区域THRESH_BINARY（暗区域THRESH_BINARY_INV），像素邻域大小（单数），偏移值调整量（单数）
        imgray2=cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,par['process_par']['area_size'],par['process_par']['bias']) # 自适应二值化11,9
    else:
        #自适应二值化,工件为黑色，图片，上限，自适应二值算法ADAPTIVE_THRESH_MEAN_C，提取亮区域THRESH_BINARY（暗区域THRESH_BINARY_INV），像素邻域大小（单数），偏移值调整量（单数）
        imgray2=cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,par['process_par']['area_size'],par['process_par']['bias']) # 自适应二值化11,9
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
    file_path = par['load_par']['file_path']
    #file_path = "yuan.jpeg"
    # 读取彩色图片
    im = cv2.imread(file_path)
    # print(np.unique(img))
    #缩小图像
    #im = cv2.resize(im, None, fx=0.25, fy=0.25)
    cut = 0#裁剪边框尺寸,用来去除边缘的噪声03-1
    #图像处理
    im, image_erzhi = tuxiangchuli(im)
    #找轮廓
    image_biankuang = biankuang(image_erzhi)
    image_biankuangH = image_biankuang.copy()
    #print(image_biankuang.shape)
    

    #R:RANSAC,H:Hough
    [imagedR, resultsR] = RansacCircleDetector(image_biankuang, par).operate()
    try:
        [imagedH, resultsH] = HoughCircleDector(image_biankuangH, par).operate()
    except:
        [imagedH, resultsH] = [image_biankuangH, [0, 0, 1]]
        print('Hough检测不到圆!')
    
    imH = im.copy()
    drawerR = Drawer(im, par)
    drawerH = Drawer(imH, par)
    
    imedR = drawerR.draw_circle(resultsR)[0]
    imedH = drawerH.draw_circle(resultsH)[0]
    

    save_name_title = file_path.split('.')[0]
    cv2.imwrite(save_name_title + '_imgedR.jpg', imagedR)
    cv2.imwrite(save_name_title + '_imR.jpg',imedR)

    cv2.imwrite(save_name_title + '_imgedH.jpg', imagedH)
    cv2.imwrite(save_name_title + '_imH.jpg',imedH)
    
    
   