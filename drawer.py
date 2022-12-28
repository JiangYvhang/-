from mine_operator import Operator

import cv2
import numpy as np

class Drawer(Operator):
    '''
    画家类

    处理显示相关

    待完成：
        统一接口
    '''
    def __init__(self, img, par: dict) -> None:
        super().__init__(img, par)
        self.par = self.par['drawer_par']


    def load_par(self, par):
        self.wait = self.par['wait']

    def operate(self):
        return [self.img, [] ]

    def show_img(self, img):
        #可调整窗口显示图像，鼠标点进图像，键盘回车，退出程序
        cv2.namedWindow('contour', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('contour', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_circle(self, results):
        
        [x, y ,r] = results
        x_int = int(x)
        y_int = int(y)

        if len(self.img.shape) == 2:
            self.img = np.stack([self.img,self.img,self.img],axis=-1)
        
        cv2.circle(self.img,(x_int, y_int),int(r),(0,0,255),2)
        # 圆心
        cv2.circle(self.img,(x_int, y_int),2,(0,255,0),3)
        cv2.putText(self.img,  '(' + str(round(x))+',' + str(round(y)) + '),r:' + str(round(r)), (0,y_int) , cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        self.show_img(self.img)

        return [ self.img, [] ]

