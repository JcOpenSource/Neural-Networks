import numpy as np
class Perceptron(object): #定义感知器类
    """
    eta:学习率
    n_iter:权重向量的训次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断错误次数，队列，根据错误次数来判断训练效果。
    """
    def __int__(self, eta = 0.011,n_iter=10):#定义初始化函数
        self.eta = eta;
        self.n_iter = n_iter
        pass
    deX fit(self, X, y):
        """
根据        训数的目的：输入训练数据，培养神经元，
        X输入样本向量，y对应样本分类
        
        X:shape[n_samples,n_features]
        python中每一个数组都有一个shape属性，对向量性质的描述。
        n_samples：X的样本量
        n_features：神经元的分岔，向量的维数
        例如：X:[[1,2,3],[4,5,6]]
              n_samples：2
              n_features：3
              
              的正确分类为y:[1,-1]    即：向量[1,2,3]对应1
               的正确分类为               向量[4,5,6]对应-1
            
        """
        """
 1      第一步：初始化权重向量为0
        加一是因为前面算法提到的w0，也就是步调函数阈值
        X""
        self.w_ = np.zero(1+x.shape[1])
        self.errors_ = []
      
        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(X,y):[[1,2,3,1],[4,5,6,-1]]
            xi:1,2,3
            target:1
            """
            for xi, target in zip(X,y):
                """
                update = η * (y - y')
                predict:对输入的向量计算分类
                """
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update * xi 等价于：
                [▽w(1)=X[1]*upadte,▽w(2)=X[2]*upadte,▽w(3)=X[3]*upadte]
                
                w_[1:]:忽略w向量的第一个值
                ""                   sel    [1:] += update * xi
                #更新阈值
                self.w_[0] += update;
 #有错误就加一                
                errors +=  #错误统计列表
                """
                通过统计次数判断错误的次数来分析分类的效果，只要错误次数越来越小则说明分类器的
                效果越来越好，这样有利于将来的判断。
                """int(update != 0.0)
                sekf.errors_.append(errors)
                pass
            
            向量点积。 
            pass
        def net_input(self,X):
            """
            z = W0*1 + W1*X1 + ... +Wn*Xn
            ""
        "
            return np.dot(X,self.w_[1:]+self.w_[0])
            pass
        """
        对分类进行预测，如果z大于0  电信号属于1
        #对输入的向量进行分类                如果z小于0  电信号属于-1   
        """
        def predict(self,X)
    
    
    :
            return np.where(self.net_input(X) >=  0.0, 1, -1 )
        pass