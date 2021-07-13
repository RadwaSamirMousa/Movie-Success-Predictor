from Preprocessing import *
from LinearRegressionModel import *
from LassoModel import *
from test import *
from Classi_PreProcessing import *
from Load import *
from LogisticRegressionModel import *
from knn import *
from SVM import *
import pickle
X_train = ' '
Y_train = ' '
x_test = ' '
y_test = ' '
model = ' '
filename = ' '

choice = int(input("Please Choose 1-classification 2-Regression "))

if choice == 1:
    #data = pd.read_csv("Movies_training_classification.csv")
    #X_train, Y_train, x_test, y_test = Classi_PreProcessing(data)
    ch = int(input("Please choose a classifer 1-Logestic regression 2-KNN 3-SVM : "))

    if ch == 1:
        try:
             load(x_test,y_test)
        except (OSError, IOError) as e:
            #model = logistic_regression(X_train,Y_train,x_test,y_test)
            model = logistic_regression(X_train,Y_train)
            load(x_test,y_test)
    elif ch == 2:
        try:
            load(x_test,y_test)
        except (OSError, IOError) as e:
            model = knn(X_train, Y_train)
            load(x_test,y_test)
            
    else:
        try:
            load(x_test,y_test)
        except (OSError, IOError) as e:
            model = SVM(X_train, Y_train)
            load(x_test,y_test)
  
   
    

if choice == 2:
    
    #data = pd.read_csv("Movies_training.csv")
    #X_train, Y_train, x_test, y_test = Data_Preprocessing(data)
    ch = int(input("Please choose 1- linear regression  , 2- Lasso Regression  "))
    if ch == 1:
        try:   
            loadRegressor(x_test, y_test)
        except (OSError, IOError) as e:
           
            model = Linear_Regression(X_train,Y_train)
            loadRegressor(x_test, y_test)
    else:
        try:
           
            loadRegressor(x_test, y_test)
        except (OSError, IOError) as e:
            model = lasso(X_train, Y_train)
            print(model)
            loadRegressor(x_test, y_test)
            
        
    

 






















