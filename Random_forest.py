#modules
from sklearn.ensemble import RandomForestClassifier
import joblib # to save and load model

class model_RandomForest : 
    def __init__(self,x_train, x_test, y_train, y_test,RandomForestclf,filename,RandomForest_list,user_predict) :
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.RandomForestclf =RandomForestClassifier()
        self.filename='RandomForest model learning.txt'
        self.RandomForest_list=[]
        self.user_predict = []

    def RandomForest_training(self):        
        #svm model training
        print(self.RandomForestclf.fit(self.x_train ,self.y_train))
        print("RandomForest Training done ✅")
        
    def RandomForest_accurecy(self):
        print(self.RandomForestclf.predict(self.x_test))
        print("the accuracy of  model is :",self.RandomForestclf.score(self.x_test,self.y_test)*100,"%")
        self.RandomForest_list=self.RandomForestclf.predict(self.x_test)
        print("__________________________________________________________________________________________________________")
        
        #svm model saving 
    def RandomForest_saving(self):
        joblib.dump(self.RandomForestclf,self.filename) 
        
        #svm model loading
    def RandomForest_loading(self):   
        loadedmodel = joblib.load(self.filename)
        self.user_predict = loadedmodel.predict(self.x_test)
        result1 = loadedmodel.score(self.x_test,self.y_test)
        print(self.user_predict)
        print(result1)
        
