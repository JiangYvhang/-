from abc import abstractmethod, ABCMeta

class Operator(metaclass = ABCMeta):
    '''
    抽象图像操作类
    
    图像操作的抽象工厂

    属性：
        img:rgb图像
        par:字典传递相关参数

    待完成：
        加载和保存类，也负责格式转换
        灵活的图像处理类，自由度更高的建造者模式
    '''
    def __init__(self, img, par:dict) -> None:
        self.img = img
        self.par = par
        #print(self.par)
        #self.load_par(self.par)
        
    
    @abstractmethod
    def load_par(self, par):
        '''
        加载相关参数

        输入：
            par字典传递相关参数:{['detection_par']:{
                                                                                            'type':circle,
                                                            }   

        输出：
            无     
        '''
        pass

    @abstractmethod
    def operate(self) :
        '''
        操作方法接口

        输入：
            img:rgb图像
            par:字典传递相关参数

        输出：
            imged:处理后的图像
            results:处理后一些结果,list
        '''
        pass