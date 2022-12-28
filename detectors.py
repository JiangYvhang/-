from mine_detector import Detector
from drawer import Drawer
from mine_operator import Operator

import numpy as np
import math
import cv2
from numpy.linalg import det
#from itertools import chain
#还需要重构，抽象出一个圆检测类，把一些常用函数放进去
#还需要考虑以后的代码复用，现在复用不是很方便
class RansacCircleDetector(Detector):
    '''
    RANSAC方法拟合圆

    使用圆的基本方程，三点定圆
    '''
    def __init__(self, img, par: dict) -> None:
        super().__init__(img, par)
        self.par = self.par['RANSAC']
        self.par_all = par

        self.load_par(self.par)
        iters_min = self.get_RANSAC_iters()#t=0.05,p=0.99,i = 36844,t=0.01,p=0.7,i = 1203977
        print('理论估算迭代参数为', iters_min, ',迭代次数要大于该值。手动设定值为', self.iters)
        if iters_min > self.iters:
            raise ValueError ('迭代次数不小于', iters_min)

    def load_par(self, par):
        '''
        加载相关参数

        输入：
            par['detection_par']['RANSAC']:字典传递相关参数{
                                                            'type':circle,
                                                            't':0.5,
                                                            'P':255,#想提取的像素值
                                                            'r_min':100,
                                                            'iters':3000,
                                                            'p':0.99,# 希望的得到正确模型的概率
                                                            'sigma':0.9,# 数据和模型之间可接受的差值
                                                            }}

        输出：
            无

        待完成：
            检测参数是否加载成功？
            是否自动为缺少参数赋值？
        '''
        self.t = par['t']
        self.P = par['P']
        self.r_min = par['r_min']
        self.iters = par['iters']
        self.p = par['p']
        self.sigma = par['sigma']

    def get_RANSAC_iters(self, n = 3, plus = 2):
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
        iters = math.log(1 - self.p) / (math.log(1 - self.t ** n))
        iters = int(iters)
        iters = iters * plus
        return iters

    def get_circle(self, p1,p2,p3):
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

    def detect(self):
        '''
        检测方法接口

        输入：
            img:二值图像
            par['detection_par']:字典传递相关参数{
                                                            'type':circle,
                                                            'algorithm':RANSAC,
                                                            't':0.5,
                                                            'P':255,
                                                            'r_min':100
                                                            }}

        输出：
            imged:处理后的图像
            results:[x, y, r]
        '''
        
        #提取255像素值的点
        points = self.dec_points()
        size = len(points)

        #用于可视化显示，在其上画彩色圆等
        img_bgr = np.stack([self.img,self.img,self.img],axis=-1)
        
        # 使用RANSAC算法估算模型
        
        # 最好模型的参数估计和内点数目
        best_x0 = 0
        best_y0 = 0
        best_r = 5
        pretotal = 0
        # 迭代最大次数，每次得到更好的估计会优化iters的数值
        #参考程序手动设置100000

        for i in range(self.iters):
            if i % 1000 == 0:
                print('已经迭代', i ,'次')
            # 随机选取3个点
            p1, p2, p3 = self.dec_random_points(3)
            
            #3点求圆方程
            #可以替换不同方案计算圆和对应内点
            #p1, p2, p3 = Point(x_1, y_1), Point(x_2, y_2), Point(x_3, y_3)
            #(x0, y0, r) = self.get_circle(p1, p2, p3)
            try:
                (x0, y0, r) = self.get_circle(p1, p2, p3)
                #print(x0,y0,r)
            except:
                continue
                
            if r > self.r_min:
                # 算出内点数目
                total_inlier = 0
                for index in range(size):
                    r_estimate2 = (points[index][0] - x0) ** 2 +  (points[index][1] - y0) **2
                    if abs(r_estimate2 - r ** 2) < self.sigma:
                        total_inlier = total_inlier + 1

                # 判断当前的模型是否比之前估算的模型好
                if total_inlier > pretotal:
                    iters = math.log(1 -self.p) / math.log(1 - pow(total_inlier / (size), 2))
                    pretotal = total_inlier
                    best_x0 = x0
                    best_y0 = y0
                    best_r = r

                # 判断是否当前模型已经符合超过一半的点
                if total_inlier > size * self.t:
                    break
            else:
                continue
        print('圆心坐标:(', best_x0, ',', best_y0, '),半径:', best_r)
       
        imged = Drawer(self.img, self.par_all).draw_circle([best_x0, best_y0, best_r])[0]

        return [imged, [best_x0, best_y0, best_r]]

class HoughCircleDector(Detector):

    def __init__(self, img, par: dict) -> None:
        super().__init__(img, par)
        self.par = self.par['Hough']
        self.par_all = par
        self.load_par(self.par)

    def load_par(self, par):
        self.bili = self.par['bili']
        self.dist_min = self.par['dist_min']
        self.can = self.par['can']
        self.acc = self.par['acc']
        self.r_min = self.par['r_min']
        self.r_max = self.par['r_max']

    def detect(self):
        
        circles = cv2.HoughCircles(self.img,cv2.HOUGH_GRADIENT,
                                                            self.bili,#图像解析的反向比例。1为原始大小，2为原始大小的一半
                                                            self.dist_min,#圆心之间的最小距离。过小会增加圆的误判，过大会丢失存在的圆
                                                            param1=self.can,#Canny检测器的高阈值
                                                            param2=self.acc,#检测阶段圆心的累加器阈值。越小的话，会增加不存在的圆；越大的话，则检测到的圆就更加接近完美的圆形
                                                            minRadius=self.r_min,#检测的最小圆的半径
                                                            maxRadius=self.r_max)#检测的最大圆的半径
        #print(np.array(circles))

        circles = np.uint16(np.around(circles))
        #print(circles.shape)
        # 因为我这里只检测一个圆，需要检测多个圆的话，就遍历circles
        #for i in circles[0, :]:
        #    cv2.circle(img_bgr, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        #    cv2.circle(img_bgr, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
        #show_img(img_bgr)
        assert len(circles)==1,f'{circles},not qualify only a circle!'
        (x,y,r) = circles[0][0]
        # input(circles[0,:])
        print('圆心坐标:(', x, ',', y, '),半径:', r)
        imged = Drawer(self.img, self.par_all).draw_circle([x, y, r])[0]
        
        return [imged, [x, y, r]]

if __name__ == '__main':
    pass