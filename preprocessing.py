#modules
from sklearn.model_selection import train_test_split # to divide data to train and test
from sklearn.preprocessing import LabelEncoder # to convert m to 1 & b to 0

class preprocessing_model :
    def __init__(self,data):
        self.data = data
        
    def data_cleaning(self) :
        # make sure that diagnosis is 1 & 0
        cols=["diagnosis"]
        le =LabelEncoder()
        for col in cols:
          self.data[col] = le.fit_transform(self.data[col])
        print(self.data[col])

        print(self.data) # print first row from datafile
        print(self.data.shape) # print number of rows & columns
         
        #null values before cleaning
        print("null values before cleaning : ")
        print(self.data.isnull().sum()) # print numbers of null values in each row
        print(self.data.duplicated().sum()) # # print numbers of duplicated values in each row
         
        #replace null values with 0 & Remove all duplicate rows
        self.data = self.data.drop(['Index'], axis = 1)
        print(self.data.shape)
        data_columns = self.data
        for col in data_columns :
            self.data[col].fillna(0, inplace = True)    
            self.data.drop_duplicates(keep='first',inplace=True)

        #null values after cleaning   
        print("null values after cleaning : ")
        print(self.data.isnull().sum()) # print numbers of null values in each row
        print(self.data.duplicated().sum()) # # print numbers of duplicated values in each row
        print(self.data) # print first row from datafile

    from sklearn.model_selection import train_test_split # to divide data to train and test
    def split(self):
        #classefication
        y = self.data.diagnosis.values
        x_data = self.data.drop(['diagnosis'], axis = 1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y, test_size =0.3,random_state=0)
        print(self.x_train.shape,self.x_test.shape)
        
    def non_split(self):
        self.y = self.data.diagnosis.values
        self.x = self.data.drop(['diagnosis'], axis = 1)
        

    
    