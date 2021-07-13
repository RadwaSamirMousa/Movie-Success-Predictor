import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle 

def lasso(X,Y):
    print("\n\n----------------------Lasso----------------------\n\n")
    
    
    model_lasso = Lasso(alpha=0.000001)
    model_lasso.fit(X, Y)
    pred_train_lasso = model_lasso.predict(X)
    print("Lasso Train Accuracy ",r2_score(Y, pred_train_lasso)*100,"%")
    print("Lasso Error ",np.sqrt(mean_squared_error(Y, pred_train_lasso))*100,"%")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X[:,1], X[:, 3], pred_train_lasso, linewidths=1, alpha=.7, edgecolor='k', s=200, c=pred_train_lasso)
    plt.show()
    
    pickle.dump(model_lasso, open('Lasso_model.pkl','wb'))
    
    return model_lasso