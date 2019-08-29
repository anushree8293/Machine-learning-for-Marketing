#!/usr/bin/env python
# coding: utf-8

# In[89]:


#import libraries
from sklearn.mixture import GaussianMixture
import csv
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from os import system
import graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# Import the function for DNN
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Import the function for DNN
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn import preprocessing
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
NNclassifier = Sequential()


# In[4]:


#Normalizing dataset
def standardize(df):
    avg = df.mean()
    stdev = df.std()
    series_standardized = (df - avg)/ stdev
    return series_standardized


# In[34]:


#data import
file1='E:\\Study\\ML\\HW_2\\bank\\bank-full.csv'
head=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']


# In[102]:


f1 = pd.read_csv(file1,names=head,skiprows=[0])
#combine into master data
frames = [f1]
x = pd.concat(frames)
x['y']=np.where(x['y']=="no",0,1)
x.describe()
data_raw=x.loc[:,['age','default','duration','y','balance']]
data_raw=pd.get_dummies(data_raw)
data_raw['age']=standardize(data_raw['age'])
data_raw['duration']=standardize(data_raw['duration'])
data_raw['balance']=standardize(data_raw['balance'])
data_raw.describe()


# In[104]:


X_train.describe()


# In[41]:


X=data_raw.loc[:,["age","duration","default_no","default_yes",'balance']]
y=data_raw.loc[:,['y']]


# In[42]:


#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=11,stratify=y)


# k means

# In[44]:


distortions = []
K = range(1,5)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_train)
    kmeanModel.fit(X_train)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[45]:


accuracy=[]
for k in range(1,6):
   kmeans = KMeans(n_clusters=k,random_state=11).fit(X_train)
   y_predict = kmeans.fit_predict(X_train)
   accuracy.append(accuracy_score(y_train,y_predict))
   print("Test Data Accuracy: %0.4f" % accuracy_score(y_train,y_predict))
   if k==2:
       print(classification_report(y_train, y_predict))


# In[46]:


kmeans = KMeans(n_clusters=2,random_state=11).fit(X_train)
y_predict = kmeans.fit_predict(X_test)
accuracy.append(accuracy_score(y_test,y_predict))
cf=confusion_matrix(y_test,y_predict)
print(cf)


# In[66]:


(7558+648)/(7558+4419+939+648)


# EM

# In[87]:


accuracy=[]
for n in range(1,6):
    gmm = GaussianMixture(n,random_state=11).fit(X_train)
    y_predict = gmm.fit_predict(X_train)
    accuracy.append(accuracy_score(y_train,y_predict)) 
    print("Train Data Accuracy: %0.4f" % accuracy_score(y_train,y_predict))
    if k==2:
        print(classification_report(y_train, y_predict))


# In[48]:


model = ExtraTreesClassifier()
model.fit(X_train, y_train.values.ravel())


dict_features = {}
for i in range(len(model.feature_importances_)):
    dict_features.update({X_train.columns[i]:100*model.feature_importances_[i]})
indices = np.argsort(model.feature_importances_)[::-1]


plt.bar(range(X_train.shape[1]),model.feature_importances_[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation='vertical')
plt.title("Feature Importance")
plt.ylabel("Gini importance")


# In[53]:


top_attributes = []
accuracy=[]
count = 1
for i in indices:
   if count < 5:
       top_attributes.append(X_train.columns[i])
       kmeans = KMeans(n_clusters=2,random_state=11).fit(X_train[top_attributes])
       y_predict_train = kmeans.fit_predict(X_train[top_attributes])
       print(i , "Train Data Accuracy: %0.4f" % accuracy_score(y_train,y_predict_train))
       accuracy.append(accuracy_score(y_train,y_predict_train))
       count+=1
   else:
       break


# In[54]:


plt.plot(range(1,5),accuracy)
plt.xlabel("Number of Vaiables")
plt.ylabel("Accuracy")
plt.title("Number of Vaiables Vs Accuracy")


# In[56]:


## Performing PCA, ICA and Randomized Projections
x=X_train[['duration','balance','age']]
xt=X_test[['duration','balance','age']]
#x= preprocessing.StandardScaler().fit_transform(x)
#xt= preprocessing.StandardScaler().fit_transform(xt)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_train =pca.fit_transform(x)
pca_test=pca.fit_transform(xt)
from sklearn.decomposition import FastICA
ica = FastICA(n_components=2)
ica_train =ica.fit_transform(x)
ica_test=ica.fit_transform(xt)
from sklearn.random_projection import GaussianRandomProjection
rca = GaussianRandomProjection(n_components=2, eps=0.1, random_state=11)
rca_train =rca.fit_transform(x)
rca_test=rca.fit_transform(xt)


# In[57]:


## K-Means Clustering Algorithm using PCA
## Train Accuracy and Train Plot 
kmeans.fit(pca_train)
y=np.array(y_train)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(pca_train)):
    predict_me = np.array(pca_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(pca_train))
yp=kmeans.predict(pca_train)
plt.scatter(pca_train[:, 0], pca_train[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[58]:


## Test Accuracy and Test Plot 
y=np.array(y_test)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(pca_test)):
    predict_me = np.array(pca_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(pca_test))
yp=kmeans.predict(pca_test)
plt.scatter(pca_test[:, 0], pca_test[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[59]:


## K-Means Clustering Algorithm using ICA
## Train Accuracy and Train Plot 
kmeans.fit(ica_train)
y=np.array(y_train)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(ica_train)):
    predict_me = np.array(ica_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(ica_train))
yp=kmeans.predict(ica_train)
plt.scatter(ica_train[:, 0], ica_train[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[61]:


## Test Accuracy and Test Plot 
y=np.array(y_test)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(ica_test)):
    predict_me = np.array(ica_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(ica_test))
yp=kmeans.predict(ica_test)
plt.scatter(ica_test[:, 0], ica_test[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[62]:


## K-Means Clustering Algorithm using RCA
## Train Accuracy and Train Plot 
kmeans.fit(rca_train)
y=np.array(y_train)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(rca_train)):
    predict_me = np.array(rca_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(rca_train))
yp=kmeans.predict(rca_train)
plt.scatter(rca_train[:, 0], rca_train[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[63]:


## Test Accuracy and Test Plot 
y=np.array(y_test)
y=y.astype(float)
correct = 0
prediction=[]
for i in range(len(rca_test)):
    predict_me = np.array(rca_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction.append(kmeans.predict(predict_me))
    if prediction[i] == y[i]:
        correct += 1
print(correct/len(rca_test))
yp=kmeans.predict(rca_test)
plt.scatter(rca_test[:, 0], rca_test[:, 1], c=yp, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans')


# In[65]:


### Performing Neural Networks with RP as it gave the best results. 

input_shap=2
classifier = Sequential()
classifier.add(Dense(10, kernel_initializer='uniform', activation= 'relu', input_shape =(input_shap,)))
classifier.add(Dense(5, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(3, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid'))
classifier.compile(optimizer= 'Adam',loss='binary_crossentropy', metrics=['accuracy'])
hist = classifier.fit(rca_train, y_train, batch_size = 10, epochs = 10)
y_predict = classifier.predict(rca_test)
y_predict = np.where(y_predict > 0.5,1,0)
y_test_array=np.array(y_test)
y_test_array= y_test_array.astype(float)
print(confusion_matrix(y_test_array,y_predict))  
print(classification_report(y_test_array,y_predict))
acc_score=accuracy_score(y_test_array,y_predict)
print(acc_score)
y_predict_train=classifier.predict(rca_train)
y_predict_train = np.where(y_predict_train > 0.5,1,0)
y_train_array=np.array(y_train)
y_train_array= y_train_array.astype(float)
print(confusion_matrix(y_train_array,y_predict_train))  
print(classification_report(y_train_array,y_predict_train))
acc_score1=accuracy_score(y_train_array,y_predict_train)
print(acc_score1)


# In[68]:


### Train and Test Error graph between all the algorithms
train_accuracy = [0.587,0.147,0.854,0.862,0.8895]
test_accuracy=[0.602,0.145,0.686,0.8602,0.8862]
Models =['KMeans','PCA+Kmeans','ICA+Kmeans','RP+Kmeans','NeuralNet+RP']
plt.plot(Models,train_accuracy,"rs--",linewidth=1,label='train')
plt.plot(Models,test_accuracy,"go--",linestyle="dashed",label='test')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.title("Train and Test Accuracy for various Models")


# In[91]:


### Cluster labels from k-means and probabilities from EM and Ran into ANN
kmeans = KMeans(n_clusters=2,random_state=11)


kmeans.fit(X_train)


# In[92]:


predictions_KMeans = kmeans.predict(X_train)
print(predictions_KMeans)


# In[101]:


Gaussian.fit(X_train)
predictions = Gaussian.predict(X_train)
probs_EM = Gaussian.predict_proba(X_train)


# In[ ]:


# Initialising the ANN
NNclassifier = Sequential()

# Adding the input layer and the first hidden layer
NNclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
NNclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))

# Adding the output layer
NNclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
NNclassifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
NNclassifier.fit(new_Df, y_train, batch_size = 10, nb_epoch = 50)


# In[ ]:


y_pred = NNclassifier.predict(new_Df)
y_pred = (y_pred > 0.5)


# In[ ]:


confusion_matrix(y_train,y_pred)


# In[ ]:


acc= round(100*accuracy_score(y_train,y_pred),2)

if acc < 50:
    y_pred = np.where(y_pred == 0, 1,0)
    
acc= round(100*accuracy_score(y_train,y_pred),2)

print("Accuracy =", acc,"%")


# In[99]:


kmeanst = KMeans(n_clusters=2,random_state=0)


kmeanst.fit(X_test)
predictions_KMeanst = kmeans.predict(X_test)
print(predictions_KMeanst)
Gaussian.fit(X_test)

predictionst = Gaussian.predict(X_test)
probs_EMt = Gaussian.predict_proba(X_test)


# In[100]:


datat = pd.DataFrame({"KM_Pred": predictions_KMeanst})
new_Dft = pd.DataFrame()
datat = pd.DataFrame({"KM_Pred": predictions_KMeanst,"EM_Prob": probs_EMt[:,0]})
new_Dft=new_Df.append(data)
new_Dft.head()


# In[ ]:


y_predt = NNclassifier.predict(new_Dft)
y_predt = (y_predt > 0.5)


# In[ ]:


acc= round(100*accuracy_score(y_test,y_predt),2)

if acc < 50:
    y_predt = np.where(y_predt == 0, 1,0)
    
acc= round(100*accuracy_score(y_test,y_predt),2)

print("Accuracy =", acc,"%")

