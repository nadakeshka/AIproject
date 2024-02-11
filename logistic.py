#module
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib # to save and load model
 
class model_LogisticRegression : 
    def __init__(self,x_train, x_test, y_train, y_test,logistic_canceer,filename,logistic_list,user_predict) :
        self.x_train = x_train 
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.logistic_canceer =LogisticRegression(solver='liblinear',C=10.0,random_state=0)
        self.filename = 'logistic model learning.txt'
        self.logistic_list= []
        self.user_predict = []
           
    def logistic_training(self):
        #model logistic---- 
        print(self.logistic_canceer.fit(self.x_train,self.y_train))
        print("Logistic Training done ✅")
        
    def logistic_accurecy(self):
        print(self.logistic_canceer.predict(self.x_test))
        print("the accuracy of logistic model is :",self.logistic_canceer.score(self.x_test,self.y_test)*100,"%")
        self.logistic_list = self.logistic_canceer.predict(self.x_test)
        print("__________________________________________________________________________________________________________")

    def logistic_saving(self):
        #logistic model saving 
        joblib.dump(self.logistic_canceer,self.filename)
        
    def logistic_loading(self):    
        #logistic loading
        loadedmodel = joblib.load(self.filename)
        self.user_predict = loadedmodel.predict(self.x_test)
        result1 = loadedmodel.score(self.x_test,self.y_test)
        print("the accuracy of decision tree model is :",result1*100,"%")
        print(self.user_predict)
        print(result1)
        print("__________________________________________________________________________________________________________")
    
         



        

    
  
    
    
    
    
    
    
    
    
    
    
    
    