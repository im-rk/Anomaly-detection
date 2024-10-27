#Anomaly detection using Unsupervised Machine learning (isolation forest).

#importing the necesary library required for this project.
#pandas are required for reading the csv file and storing it in an array
import pandas as pd

#matplotlib are required for plotting 
import matplotlib.pyplot as plt

#sklearn.ensemble is the main lib that contains isolation forest which comes under unspervised machine learning
from sklearn.ensemble import IsolationForest

#
import numpy as np

#-----------------------------------------------------------------------------------------------------
#by randomly generated data
random_seed=np.random.RandomState(12)
x_train=0.5*random_seed.randn(500,2)
x_train=np.r_[x_train+3,x_train]
x_train=pd.DataFrame(x_train,columns=["x","y"])
print(x_train)


x_test=0.5*random_seed.randn(500,2)
x_test=np.r_[x_test+3,x_test]
x_test=pd.DataFrame(x_test,columns=["x","y"])
print(x_test)

x_outliers=random_seed.uniform(low=-5,high=5,size=(50,2))
x_outliers=pd.DataFrame(x_outliers,columns=["x","y"])
print(x_outliers)

p1=plt.scatter(x_train.x,x_train.y,c="white",s=50,edgecolors="black")
p2=plt.scatter(x_test.x,x_test.y,c="green",s=50,edgecolors="black")
p3=plt.scatter(x_outliers.x,x_outliers.y,c="blue",s=50,edgecolors="black")
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.legend([p1,p2,p3],["training set","normal testing set","anomalous testing set"],loc="lower right")
plt.show()

clf=IsolationForest()
clf.fit(x_train)
train_pred=clf.predict(x_train)
test_pred=clf.predict(x_test)
outlier_pred=clf.predict(x_outliers)

x_outliers=x_outliers.assign(pred=outlier_pred)
p1=plt.scatter(x_train.x,x_train.y,c="white",s=50,edgecolors="black")
p2=plt.scatter(x_outliers.loc[x_outliers.pred==-1,["x"]],x_outliers.loc[x_outliers.pred==-1,["y"]],c="blue",s=50,edgecolors="black")
p3=plt.scatter(x_outliers.loc[x_outliers.pred==1,["x"]],x_outliers.loc[x_outliers.pred==1,["y"]],c="red",s=50,edgecolors="black")
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.legend([p1,p2,p3],["training observations","detected outliners","incorrect labeled outliers"],loc="lower right")
plt.show()


#--------------------------------------------------------------------------------------------------------------
#reading the data .csv file (dataset)
data=pd.read_csv("annual.csv")
data = data[data['Source'] == 'gcag']
data=data.set_index("Year")
data=data[["Mean"]]
#print(data)

#plotting the data using pyplot by .scatter
plt.scatter(data.index,data[["Mean"]])
plt.xlabel("Year")
plt.ylabel("Mean")
plt.title("Mean over years")
plt.show()  #for showing the plot

#Isolation forest is the technique used for finding the anomalies.
#The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples.
iff=IsolationForest(contamination=0.15)    # isolation forest algo.
iff.fit(data)

#Return the anomaly score of each sample using the IsolationForest algorithm.
pred_ictions=iff.predict(data)             
print(pred_ictions)

#We are storing the index of the predictions which are less than zero or equal to -1 in an array named abnormal_index.
abnormal_index=np.where(pred_ictions<0)

x=data.values
print(x)

plt.scatter(data.index,data["Mean"],label="Normal Data",color="b")
plt.scatter(data.index[abnormal_index],data['Mean'].iloc[abnormal_index],label="Anomalies",edgecolors='r',color="none")
plt.xlabel("year")
plt.ylabel("mean")
plt.legend()
plt.title("Mean over years with Anomalies Highlighted")
plt.show()


