#modules
from preprocessing import preprocessing_model
from logistic import model_LogisticRegression
from decisiontree import model_DecisionTree
from svm import model_svm


class uploading:
    def __init__(self,new_data):
        self.new_data = new_data
        
  
    def uploading(self):        
        #preprocessing object
        user_preprocessing1 = preprocessing_model(self.new_data)   
        user_preprocessing1.data_cleaning()
        user_preprocessing1.non_split()
        
        #logistic model object
        user_logistic1 = model_LogisticRegression(0, user_preprocessing1.x,0, user_preprocessing1.y, 0, "",[],[])
        user_logistic1.logistic_loading()
        
        #decisiontree model object
        user_Decisiontree1 = model_DecisionTree(0, user_preprocessing1.x,0, user_preprocessing1.y, 0, "",[],[])
        user_Decisiontree1.decisiontree_loading()
        
        #svm model object
        user_svm1 = model_svm(0, user_preprocessing1.x,0, user_preprocessing1.y, 0, "",[],[])
        user_svm1.svm_loading()
        
        
        
        voting_list=[]
        for x in range(len(user_logistic1.user_predict)):
            voting_list.append(user_logistic1.user_predict[x] + user_Decisiontree1.user_predict[x] + user_svm1.user_predict[x]) 
            if voting_list[x]==3 or voting_list[x]==2 :
                voting_list.pop()
                voting_list.append("Malignant tumer!")
            else: 
                voting_list.pop()
                voting_list.append("Benign tumer!")           
        print (voting_list)




