from mine_operator import Operator
from abc import abstractmethod, ABCMeta

import numpy as np
import random

class Detector(Operator, metaclass = ABCMeta):
    '''
    检测工厂

    属性：
        img:rgb图像
        par:字典传递相关参数
    '''
    def __init__(self, img, par: dict) -> None:
        super().__init__(img, par)
        self.par = self.par['detection_par']
        self.P = self.par['P']
        #self.load_par(self.par)
        #self.load_par(self, par['detection_par'])
    
    def operate(self) :
        '''
        操作方法接口

        输入：
            img:二值图像
            par:字典传递相关参数{'detection_par':{
                                                            'type':circle,
                                                            ''
                                                            }}

        输出：
            imged:处理后的图像
            results:处理后一些结果,list
        '''

        if len(self.img.shape) != 2:
            raise TypeError('待检测图像需为二维图像') 

        [imaged, results] = self.detect()
        
        return imaged, results

    @abstractmethod
    def detect(self):
        '''
        检测方法接口

        输入：
            img:二值图像

        输出：
            imged:处理后的图像
            results:处理后一些结果,list,[imged, results([x,y,r])]
        '''
        pass

    @abstractmethod
    def load_par(self, par):
        '''
        加载相关参数

        输入：
            par['detection_par']:字典传递相关参数{
                                                            'type':circle,
                                                            ''   

        输出：
            无     
        '''
        pass

    def dec_points(self):
        '''
        检测图片中像素值为p的点

        输入：
            img:二维的二值化图像
            p:int 要检测的像素值

        输出：
            列表[ [x1, y1], [x2, y2], [x3, y3] ]
        '''
        img = np.array(self.img)
        points = np.where(img == self.P)
        #print(points[0].size)
        points = [ [points[0][i], points[1][i] ] for i in range(points[0].size) ] 
        #print(points)
        return points

    def dec_random_points(self, num:int):
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
            img_points = self.dec_points()
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

if __name__ == '__main':
    pass