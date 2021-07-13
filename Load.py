import pickle
from sklearn import metrics
import time 
import pandas as pd
from Classi_PreProcessing import *
from Preprocessing import *
def load(x_test,y_test):
    
    
    data = pd.read_csv("test2.csv")
    x_test, y_test = Classi_PreProcessing(data)
     
    
    
    model = pickle.load(open('SVM_model.pkl','rb'))
    
    start_time = time.time()
    print("Test accuracy  ", round(model.score(x_test, y_test) * 100), "%")
    end_time = time.time()
    total_test_time = end_time - start_time
    print("total test time ",total_test_time)
    
    
    
def loadRegressor(x_test,y_test):
    
    
    data = pd.read_csv("test3.csv")
    x_test, y_test = Data_Preprocessing(data)
    
    model = pickle.load(open('LinearRegression_model.pkl','rb'))
    print("x_test is " , x_test.shape)
    print("y_test is ", y_test.shape)
    
    
    #x_test = x_test.reshape(50,3694)
    
    
  
    
    pred_test = model.predict(x_test)
    

    
    
    
    
    print("R2 Score ", abs(metrics.r2_score(y_test,pred_test)))
    print('MSE ', metrics.mean_squared_error(y_test, pred_test))
    
    #print("Test accuracy  ", round(model.score(x_test, y_test) * 100), "%")
     

  

