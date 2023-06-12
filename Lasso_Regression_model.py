import numpy as np
class Lasso_Regression:
  def __init__(self,Learning_Rate,no_of_epochs,Lambda_parameter):
    self.Learning_Rate=Learning_Rate
    self.no_of_epochs=no_of_epochs
    self.Lambda_Parameter=Lambda_parameter

  def fit(self,X,Y):
    self.rows,self.cols=X.Shape

    self.w=np.zeros(self.cols)
    self.b=0
    self.X=X
    self.Y=Y

    for i in range(self.no_of_epochs):
      self.update_weights()
    


  def update_weights(self):
    Y_pred=self.predict(self.X)
    dw=np.zeros(self.cols)
    db=-2*np.sum(self.Y-Y_pred)/self.rows

    for i in range(self.cols):
      if self.w[i]>0:
        dw[i]=((self.X[:,i].dot(self.Y-Y_pred))+self.Lambda_Parameter)*(-2)/self.rows

      else:
        dw[i]=((self.X[:,i].dot(self.Y-Y_pred))-self.Lambda_Parameter)*(-2)/self.rows


    self.w=self.w-self.Learning_Rate*dw
    self.b=self.b-self.Learning_Rate*db


  def predict(self,X):
    return X.dot(self.w)+self.b


