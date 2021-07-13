from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import time



def logistic_regression(x_train,y_train):


    print("\n\n----------------------Logistic Regression----------------------\n\n")


    logistic = LogisticRegression(max_iter=1000)
    #logistic = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    logistic_start_time = time.time()
    logistic.fit(x_train, y_train)
    logistic_end_time = time.time()
    logistic_training_time = logistic_end_time - logistic_start_time
    
    pred_train = logistic.predict(x_train)
    
    
    print("total training time  ", logistic_training_time)
    print("Training accuracy  ", round(logistic.score(x_train, y_train) * 100), "%")
    
    
    #filename = 'Logestic_model.sav'
    #pickle.dump(logistic, open(filename, 'wb'))
    
    # Saving model to disk
    pickle.dump(logistic, open('LogisticRegression_model.pkl','wb'))
    ##################################################################3
    #model = pickle.load(open('LogisticRegression_model.pkl','rb'))
    #Make Prediction
    #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(x_test,y_test)))
    #print('Test Accuracy is',round(model.score(x_test,y_test)*100), "%")




    return logistic