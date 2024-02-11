#modules
from sklearn.svm import SVC
import joblib # to save and load model

class model_svm : 
    def __init__(self,x_train, x_test, y_train, y_test,svm_canceer,filename,svm_list,user_predict) :
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.svm_canceer =SVC(kernel='linear', C=0.01)# kernel because we have 30-d , C for overfitting
        self.filename='svm model learning.txt'
        self.svm_list=[]
        self.user_predict = []

    def svm_training(self):        
        #svm model training
        print(self.svm_canceer.fit(self.x_train ,self.y_train))
        print("SVM Training done ✅")
        
    def svm_accurecy(self):
        print(self.svm_canceer.predict(self.x_test))
        print("the accuracy of svm model is :",self.svm_canceer.score(self.x_test,self.y_test)*100,"%")
        self.svm_list=self.svm_canceer.predict(self.x_test)
        print("__________________________________________________________________________________________________________")
        
        #svm model saving 
    def svm_saving(self):
        joblib.dump(self.svm_canceer,self.filename) 
        
        #svm model loading
    def svm_loading(self):   
        loadedmodel = joblib.load(self.filename)
        self.user_predict = loadedmodel.predict(self.x_test)
        result1 = loadedmodel.score(self.x_test,self.y_test)
        print("the accuracy of svm tree model is :", result1 * 100 ,"%")
        print(self.user_predict)
        print(result1)
        print("__________________________________________________________________________________________________________")
        
        
        


        

        

    
    
    