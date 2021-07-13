from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle


def Linear_Regression(X,Y):
    print("\n\n----------------------Linear Regression----------------------\n\n")
    
    
    cls = linear_model.LinearRegression()
    cls.fit(X,Y) #Fit method is used for fitting your training data into the model
    prediction= cls.predict(X)
    print('Co-efficient of linear regression',cls.coef_)
    print('Intercept of linear regression model',cls.intercept_)
    print('Linear Regression Train Accuracy ',metrics.r2_score(Y,prediction)*100 , "%")
    print('Linear Regression Mean Square Error', metrics.mean_squared_error(Y, prediction)*100,"%")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,1],X[:,3],prediction,linewidths=1, alpha=.7,edgecolor='k',s=200,c=prediction)
    plt.show()
    
    
    pickle.dump(cls, open('LinearRegression_model.pkl','wb'))

    return cls




