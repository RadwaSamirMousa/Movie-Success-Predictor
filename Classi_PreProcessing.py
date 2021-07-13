import pandas as pd
import numpy as np
from pandas import unique
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statistics import mode
from sklearn import utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# Language => 998 labels
# Genres => 1588 labels
# Country => 1104 labels

'''def enc(name, lst):
    for label in labels:
        df[variable + '_' + label] = np.where(data[variable] == label, 1, 0)'''


#curse of dimensionality !!! E , A , F  (1,2,3,1,2,3) 

def Encode_CatergoricalFeatures(data ,feature,lst):
    j=0
    for i in lst:
        data[feature] = data[feature].replace(i,j+1)
        j+=1


def Classi_PreProcessing(data):


    data = pd.read_csv("Movies_training_classification.csv")


    Encode_CatergoricalFeatures(data,"Language", data["Language"].unique())
    Encode_CatergoricalFeatures(data,"Genres", data["Genres"].unique())
    Encode_CatergoricalFeatures(data, "Country", data["Country"].unique())


    '''LangFeature = [x for x in data["Language"].value_counts(ascending=False).index]
    one_hot_enc(data, 'Language', LangFeature, data)

    GenresFeature = [x for x in data["Genres"].value_counts(ascending=False).index]
    one_hot_enc(data, 'Genres', GenresFeature, data)

    CountryFeature = [x for x in data["Country"].value_counts(ascending=False).index]
    one_hot_enc(data, 'Country', CountryFeature, data)

    #titlesFeature = [x for x in data["Title"].value_counts(ascending=False).index]
    #one_hot_enc(data, 'Title', titlesFeature, data)'''

    f = data["Rotten Tomatoes"].str.replace('%', ' ')
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    f = df.fillna(np.nanmean(f)) # fill with mean of column.
    data["Rotten Tomatoes"] = f.astype(np.float)

    f = data["Age"].str.replace('all', '0')
    f = f.str.replace('+', '')
    df = pd.DataFrame(f)
    f = np.array(df)
    f = f.astype(np.float)
    f = df.fillna(np.nanmean(f))
    
    #f = df.fillna(f.mode().iloc[0]) #most frequent 
    
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

    ##f = data["IMDb"]
    ##df = pd.DataFrame(f)
    ##f = np.array(df)
    ##f = f.astype(np.float)
    ##data["IMDb"] = df.fillna(np.nanmean(f).round(1))


    ##f = data["rate"].str.replace("Low", '-1')
    ##f = data["rate"].str.replace("Intermediate", '0')
    ##f = data["rate"].str.replace("High", '1')
    ##df = pd.DataFrame(f)
    ##f = np.array(df)
    ##f = f.astype(np.float)
    ##data["rate"] = df.fillna('0')
    ##data["rate"] = f.astype(np.int)
    j = 0
    lst = ['Low', 'Intermediate', 'High']
    for i in lst:
        data['rate'] = data['rate'].replace(i, j + 1)
        j += 1
    data["rate"] = df.fillna(1)
    data["rate"] = df.astype(np.int)
##
    data.rename(columns={"Netflix": "is_Streamed"}, inplace=True)
    #data["is_Streamed"] = data["is_Streamed"]  | data["Hulu"] | data["Prime Video"] | data["Disney+"]
    
    
    #Aggregation 
    data["is_Streamed"] = data["is_Streamed"] + data["Hulu"] + data["Prime Video"] + data["Disney+"]
   
    
    
    Y = data["rate"]
    
    Dropped_Cols = ["Title","Directors","Type", "Hulu", "Prime Video", 
                    "Disney+","Language","Country","Runtime"]
    data.drop(Dropped_Cols, inplace=True, axis=1)
 


    #print(data.corr())
    
    
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    Y = Y.values.reshape(11743, )
    

   
    
    data = np.array(data)
    Y = np.array(Y)
    #shuffle(data, Y)
    return data , Y



    # 80% train , 20% test 
    #x_train, x_test, y_train, y_test = train_test_split(
    #data, Y, test_size = 0.2, random_state = 42)

    #return x_train, y_train, x_test, y_test