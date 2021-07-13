import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle
import time

def knn (X_train,y_train):
 print("\n\n---------------------KNN-----------------------\n\n")
 error = []
 knn = ' '
 pred_i = ' '
# Calculating error for K values between 1 and 40
 for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn_start_time = time.time()
    knn.fit(X_train, y_train)
    knn_end_time = time.time()
    pred_i = knn.predict(X_train)
    error.append(np.mean(pred_i != y_train))
    
 knn_training_time = knn_end_time - knn_start_time  
 print("total training time  ",knn_training_time) 
 print("Training accuracy  ", round(knn.score(X_train, y_train) * 100), "%")
 
 plt.figure(figsize=(12, 6))
 plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
 plt.title('Error Rate K Value')
 plt.xlabel('K Value')
 plt.ylabel('Mean Error')
 plt.show()
 
 pickle.dump(knn, open('KNN_model.pkl','wb'))




 return knn