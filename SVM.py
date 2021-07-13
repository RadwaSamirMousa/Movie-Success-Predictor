from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
import time

def SVM (X, y):

    
    print("\n\n--------------------------SVM------------------------\n\n")

    classifier = SVC(C = 1000, kernel='rbf' ,gamma=0.1)
    SVM_start_time = time.time()
    classifier.fit(X, y)
    SVM_end_time = time.time()
    
    SVM_training_time = SVM_end_time - SVM_start_time
    

    print("total training time  ", SVM_training_time)
    print("Training accuracy  ", round(classifier.score(X, y)*100), '%')
    
    
    '''classifier = SVC(C = 0.1, kernel='rbf' ,gamma=0.1)
    classifier.fit(X, y)
    print('Training accuracy is', round(classifier.score(X, y)*100), '%')
    
    
    classifier = SVC(C = 1, kernel='rbf' ,gamma=0.1)
    classifier.fit(X, y)
    print('Training accuracy is', round(classifier.score(X, y)*100), '%')
    
    
    classifier = SVC(C = 1, kernel='linear' ,gamma=0.1)
    classifier.fit(X, y)
    print('Training accuracy is', round(classifier.score(X, y)*100), '%')'''
    
    
    
    pickle.dump(classifier, open('SVM_model.pkl','wb'))



    return classifier