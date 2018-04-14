# KNN is a unsupervised machine learning algorithm that check for neighbours to decide where to locate 
# the given data point . the No of neighbours is set to odd because if it will be even then the algorithm will get 
# equal response from to of the data labels to break it we use odd neighbors.

#initialization of basic libraries that we will need 

# -------------------------------------------------------------------------------

import numpy as np                                               #numpy for basic matrix multiplication.
import matplotlib.pyplot as plt                                  #matplotlib for plotting graphs.
import pandas as pd                                              #pandas to read and manipulate data files.
from sklearn.model_selection import train_test_split             #sklearn to import training and splitting modules and our classifier.  


# In[9]:

#getting the iris dataset from our Downloads folder first we have to download the dataset from https://www.kaggle.com/uciml/iris/data
Iris = pd.read_csv('Downloads/Iris.csv')


# displaying the first few lines of our data to check if it is correct.
Iris.head()


# generating a dictionary of unique mapping of id to specie


iris_lookup = dict(zip(Iris.Id.unique(),Iris.Species.unique()))


# printing our dictionary:


print(iris_lookup)


#setting X for our training data:


X = Iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 


#printing our data for X:

print (X)


#setting Y for out training data:


Y = Iris['Species']


#printing our data for y:


print (Y)


# Splitting out data into training and testing data:


X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state =0)


# displaying out X_train:


print (X_train)


#importing out algorithm for classification here we are using KNN (K nearest neighbours):
from sklearn.neighbors import KNeighborsClassifier


# setting n_neighbors to 7 to check 7 neighbouring points from our plotted point:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


#checking the accuracy of our algorithm:


knn.score(X_test,y_test)


# predicting type of flower giving our algorithm new data:


irispredict = knn.predict([6,4,5,1])
irispredict[0]

#-----------------------------------------------------------------------

