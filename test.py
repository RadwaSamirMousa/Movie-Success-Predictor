from sklearn import metrics

def test(model,x_test,y_test):

    pred_test = model.predict(x_test)
    print("Testing_Accuracy ", abs(metrics.r2_score(y_test,pred_test)* 100), "%")
    print('Testing_Error ', metrics.mean_squared_error(y_test, pred_test))