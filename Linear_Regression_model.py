import numpy as np 
class Linear_Regression:
        def __init__(self,Learning_rate,no_of_iterations):
            self.Learning_rate=Learning_rate
            self.no_of_iterations=no_of_iterations
             
        
        def fit(self,X,Y):
            
            #first check dimensionality
            
            if X.shape[0] != Y.shape[0]:
                raise ValueError("Number of samples in X and Y should match.")
            
            self.m,self.n=X.shape #no. of rows and columns of features
            
            #initialising wt. and bias
            
            self.w=np.zeros(self.n)  #since the no. of features may not be 1
            self.b=0
            self.X=X
            self.Y=Y
            
            #implementing gradient descent
            
            for i in range(self.no_of_iterations):
                self.update_weights()
                
            # return self.w, self.b
        
        def update_weights(self ):
             
            Y_prediction=self.predict(self.X)
            
            #calculate gradients
            
            dw=-(2*(self.X.T).dot(self.Y-Y_prediction))/self.m #dw for each feature
            db=-(2*np.sum(self.Y-Y_prediction))/self.m
            
            #update the weights
            
            self.w=self.w-self.Learning_rate*dw
            self.b=self.b-self.Learning_rate*db
            
            
        
        def predict(self,X):
            return X.dot(self.w)+self.b   #Y=WX+b
        
        