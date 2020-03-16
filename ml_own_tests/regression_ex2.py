import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt


xs=np.array([1,2,3,4,5,6], dtype=np.float64)#per linear regression importa el datatype
ys=np.array([5,4,6,5,6,7], dtype=np.float64)

plt.scatter(xs,ys)
plt.show()
