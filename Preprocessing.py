import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#Language => 998 labels
#Country => 1104 labels
#Generes => 1588 labels

#lan A , E , F (1,0,0) (0,1,0) (0,0,1)

def one_hot_enc(df, variable, labels,data):
    for label in labels:
        df[variable + '_' + label] = np.where(data[variable] == label, 1, 0)


def Data_Preprocessing(data):
    LangFeature = [x for x in data["Language"].value_counts(ascending=False).index]
    
    #data = pd.read_csv("Movies_training.csv")
    data = pd.read_csv("test3.csv")
    
    
    one_hot_enc(data,'Language',LangFeature,data)
    GenresFeature = [x for x in data["Genres"].value_counts(ascending=False).index]
    one_hot_enc(data,'Genres',GenresFeature,data)
    CountryFeature = [x for x in data["Country"].value_counts(ascending=False).index]
    one_hot_enc(data,'Country',CountryFeature,data)
    titlesFeature = [x for x in data["Title"].value_counts(ascending=False).index]
    #one_hot_enc(data, 'Title', titlesFeature, data)

    f = data["Rotten Tomatoes"].str.replace('%', ' ')
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    f = df.fillna(np.nanmean(f))
    data["Rotten Tomatoes"] = f.astype(np.float)

    f = data["Age"].str.replace('all', '0')
    f = f.str.replace('+', '')
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    f = df.fillna(np.nanmean(f))
    data["Age"] = f.astype(np.int)

    f = data["Runtime"]
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    f = df.fillna(np.nanmean(f))
    data["Runtime"] = f.astype(np.float)

    f = data["Year"]
    df = pd.DataFrame(f)
    f = np.array(df)
    data["Year"] = f.astype(np.float)

    f = data["IMDb"]
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    data["IMDb"] = df.fillna(np.nanmean(f).round(1))

    data.rename(columns={"Netflix":"is_Streamed"},inplace=True)
    data["is_Streamed"] = data["is_Streamed"] | data["Hulu"] | data["Prime Video"] | data["Disney+"]
    Y = data["IMDb"]
    Dropped_Cols = ["Title","Age","Directors","Type","Country","Hulu","Prime Video","Disney+","IMDb","Genres","Language"]
    data.drop(Dropped_Cols,inplace=True,axis=1)
    #print(data)

    '''scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    scaler = MinMaxScaler()
    Y = scaler.fit_transform(Y.values.reshape((data.shape[0],1)))
    data = np.array(data)'''
    


    '''x_train = data[0:8220,:]
    Y = np.array(Y)
    y_train = Y[0:8220]

    x_test = data[8220:,:]
    y_test = Y[8220:]
    return x_train , y_train ,x_test,y_test'''


    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    Y = Y.values.reshape(data.shape[0], )
 
    
 
    data = np.array(data)
    Y = np.array(Y)
    #shuffle(data, Y)
    print("data is : ",data.shape)
    print("Y is :",Y.shape)
    return data , Y






