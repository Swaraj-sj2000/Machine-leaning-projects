import numpy as np
class Logistic_Regression:
    def __init__(self,Learning_Rate,no_of_iterations):
        self.Learning_Rate=Learning_Rate
        self.no_of_iterations=no_of_iterations



    def fit(self,X,Y):
        self.rows,self.cols=X.shape   #rows=no of data points & cols=no of features
        self.w=np.zeros(self.cols)
        self.b=0
        self.X=X
        self.Y=Y

        for i in range(self.no_of_iterations):
            self.update_weight()


    def update_weight(self):
        z=self.X.dot(self.w)+self.b
        Y_hat=1/(1+np.exp(-z))


        dw=np.dot(self.X.T,(Y_hat-self.Y))/self.rows
        db=np.sum(Y_hat-self.Y)/self.rows

        self.w=self.w-self.Learning_Rate*dw
        self.b=self.b-self.Learning_Rate*db


    def predict(self,X):
        Y_predict=1/(1+np.exp(-(X.dot(self.w)+self.b)))
        Y_predict=np.where(Y_predict>0.5,1,0)
        return Y_predict

