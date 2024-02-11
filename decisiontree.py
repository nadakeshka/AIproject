#modules
from sklearn.tree import DecisionTreeClassifier
import joblib # to save and load model
    
class model_DecisionTree :
    def __init__(self,x_train, x_test, y_train, y_test,decision_canceer,filename,decisiontree_list,user_predict) :
        self.x_train = x_train 
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.decision_canceer =DecisionTreeClassifier(max_depth=2)
        self.filename = 'decision tree model learning.txt'
        self.decisiontree_list=[]
        self.user_predict = []
        
    def decisiontree_training(self):
        #model decisiontree
        print(self.decision_canceer.fit(self.x_train,self.y_train))
        print("Decision tree Training done ✅")
        
    def decisiontree_accurecy(self):
        print(self.decision_canceer.predict(self.x_test))
        print("the accuracy of decision tree model is :",self.decision_canceer.score(self.x_test,self.y_test)*100,"%")
        self.decisiontree_list=self.decision_canceer.predict(self.x_test)
        print("__________________________________________________________________________________________________________")

    def decisiontree_saving(self):
        #disision_canceer model saving 
        joblib.dump(self.decision_canceer,self.filename)
        
    def decisiontree_loading(self):
        #disision_canceerloading
        loadedmodel = joblib.load(self.filename)
        self.user_predict = loadedmodel.predict(self.x_test)
        result1 = loadedmodel.score(self.x_test,self.y_test)
        print("the accuracy of decision tree model is :",result1*100,"%")
        print(self.user_predict)
        print(result1)
        print("__________________________________________________________________________________________________________")
        

        

    
    
    
    
    
    
    
