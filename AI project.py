#modules

# import classes 
from preprocessing import preprocessing_model
from logistic import model_LogisticRegression
from decisiontree import model_DecisionTree
from svm import model_svm
from uploadtest import uploading

    
import pandas as pd
#First Input
print("please enter data ")
user_path = input()
data = pd.read_csv(user_path)

#object from preprocessing
user_preprocessing = preprocessing_model(data)
user_preprocessing.data_cleaning() 
user_preprocessing.split()
user_preprocessing.non_split()


#logistic model object
user_logistic = model_LogisticRegression(user_preprocessing.x_train, user_preprocessing.x_test, user_preprocessing.y_train, user_preprocessing.y_test, 0," ",[],[])
user_logistic.logistic_training()
user_logistic.logistic_accurecy()
user_logistic.logistic_saving()
user_logistic.graph()


#decisiontree model object
user_Decisiontree = model_DecisionTree(user_preprocessing.x_train, user_preprocessing.x_test, user_preprocessing.y_train, user_preprocessing.y_test, 0, "",[],[])
user_Decisiontree.decisiontree_training()
user_Decisiontree.decisiontree_accurecy()
user_Decisiontree.decisiontree_saving()

#svm model object
user_svm = model_svm(user_preprocessing.x_train, user_preprocessing.x_test, user_preprocessing.y_train, user_preprocessing.y_test, 0, "",[],[])
user_svm.svm_training()
user_svm.svm_accurecy()
user_svm.svm_saving()
      
#Voting function
def voting():
    voting_list=[]
    for x in range(len(user_logistic.logistic_list)):
        voting_list.append(user_logistic.logistic_list[x] + user_Decisiontree.decisiontree_list[x] + user_svm.svm_list[x]) 
        if voting_list[x]==3 or voting_list[x]==2 :
            voting_list.pop()
            voting_list.append("Malignant tumer!")
        else: 
            voting_list.pop()
            voting_list.append("Benign tumer!")       
    print("Voting Results :")     
    print (voting_list)  
voting()

#Second Input
print("please enter data ")
new_user = input()
new_data = pd.read_csv(new_user)
    
uploadtest_user = uploading(new_data)
uploadtest_user.uploading()



    
        
        
    










  
    
    




