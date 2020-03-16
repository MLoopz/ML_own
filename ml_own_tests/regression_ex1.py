import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
df = quandl.get('WIKI/GOOGL') #Repo para datos
#print(df.head())#datos de las acciones, precio de apertura, preci de cireerre, maximos y minimos.
#df = [df['Adj. Open'], df['Adj. High'],df['Adj. Low'],df['Adj. Close'], df['Adj. Volume']]
#df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df["HL_pctge"]= (df ['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df["change_pctge"]= (df ['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[["Adj. Close","HL_pctge","change_pctge","Adj. Volume"]]
#print(df.head())
prediction_col = "Adj. Close"
df.fillna(-99999, inplace=True)#change nan for -999 to avoid errors

prediction_out=  int(math.ceil(0.01*len(df))) #redondea aun valor con el 0.5% de datos
df["label"]= df[prediction_col].shift(prediction_out)
df.dropna(inplace=True)

#va bien normalizar las features ente -1 y 1
X=np.array(df.drop(["label"],1))
X=preprocessing.scale(X)
y=np.array(df["label"])

X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)

#n_jobs = parallelizar threads. Default regression=1.-1 max of processor
clf= LinearRegression(n_jobs=-1) #change the alogithm to svm.SVR() or others
clf.fit(X_train,y_train)
accuracy=clf.score(X_test, y_test)

print(prediction_out)
print(accuracy)

#x.plot() to build a graphic
#import pickle -> to save a classifier trained as an object, aand jst load without training it
'''with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf,f)

pickle_in= open('linearregregresion.pickle', 'wb')
clf=pickle.load(pickle_in)'''