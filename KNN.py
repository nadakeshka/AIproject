#modules
from sklearn.neighbors import KNeighborsClassifier
import joblib # to save and load model

class model_knn : 
    def __init__(self,x_train, x_test, y_train, y_test,knnclf,filename,knn_list,user_predict) :
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.knnclf = KNeighborsClassifier()
        self.filename='KNN model learning.txt'
        self.knn_list=[]
        self.user_predict = []

    def knn_training(self):        
        #svm model training
        print(self.knnclf.fit(self.x_train ,self.y_train))
        print("KNN Training done ✅")
        
    def knn_accurecy(self):
        print(self.knnclf.predict(self.x_test))
        print("the accuracy of KNN model is :",self.knnclf.score(self.x_test,self.y_test)*100,"%")
        self.knn_list=self.knnclf.predict(self.x_test)
        print("__________________________________________________________________________________________________________")
        
        #svm model saving 
    def knn_saving(self):
        joblib.dump(self.knnclf,self.filename) 
        
        #svm model loading
    def knn_loading(self):   
        loadedmodel = joblib.load(self.filename)
        self.user_predict = loadedmodel.predict(self.x_test)
        result1 = loadedmodel.score(self.x_test,self.y_test)
        print(self.user_predict)
        print(result1)