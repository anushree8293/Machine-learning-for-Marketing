#!/usr/bin/env python
# coding: utf-8

# In[82]:


#Start and connect to a local h2o cluster
import h2o
h2o.init(nthreads=-1)
h2o.connect(ip="10.21.25.193",port=54321)


# In[141]:


#import libraries
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

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Import the function for DNN
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch


# In[94]:


#Normalizing dataset
def standardize(df):
    avg = df.mean()
    stdev = df.std()
    series_standardized = (df - avg)/ stdev
    return series_standardized


# In[95]:


#data import
file1='E:\\Study\\ML\\HW_2\\bank\\bank-full.csv'
head=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']


# In[96]:


f1 = pd.read_csv(file1,names=head,skiprows=[0])
#combine into master data
frames = [f1]
x = pd.concat(frames)
x['y']=np.where(x['y']=="no",0,1)
#x.describe()
selectfeatures=x.loc[:,['age','default','duration','y']]
selectfeatures=pd.get_dummies(selectfeatures)
selectfeatures['age']=standardize(selectfeatures['age'])
selectfeatures['duration']=standardize(selectfeatures['duration'])
selectfeatures.describe()


# In[87]:





# In[98]:


hf_data = h2o.H2OFrame(selectfeatures)
hf_data['y'].table()


# In[99]:


hf_data['default_no']=hf_data['default_no'].asfactor()
hf_data['default_yes']=hf_data['default_yes'].asfactor()
hf_data.head()


# In[100]:


features = list(hf_data.columns) # we want to use all the information
features.remove('y')    # we need to exclude the target 'quality' (otherwise there is nothing to predict)
features


# In[101]:


# Split the H2O data frame into training/test sets
data_split = hf_data.split_frame(ratios = [0.7,0.15], seed = 1234)

fb_train = data_split[0] # using 70% for training
fb_valid = data_split[1]  
fb_test = data_split[2]  
fb_train_x = fb_train.drop('y')
fb_train_y = fb_train['y']
fb_train.shape


# In[102]:


fb_valid_x = fb_valid.drop('y')
fb_valid_y = fb_valid['y']
fb_valid.shape


# In[103]:


fb_test_x = fb_test.drop('y')
fb_test_y = fb_test['y']
fb_test.shape


# In[120]:


hyper_params = {'activation': ['tanh', 'rectifier','maxout'],
                'hidden': [[3],[3,3],[3,3,3]],
                'epochs': [20,50,75],
                'epsilon': [1e-10, 1e-8, 1e-6, 1e-4],
                'rho' : [0.9, 0.99, 0.999],
                'rate' : [0, 0.01, 0.005, 0.001],
                'rate_annealing': [1e-8, 1e-7, 1e-6],
                'l1': [0, 1e-3, 1e-5],
                'l2': [0, 1e-3, 1e-5],
                'momentum_start' : [0, 0.5],
                'momentum_stable': [0.99, 0.5, 0],
                'input_dropout_ratio': [0, 0.1, 0.2],
                'max_w2' : [10, 100, 1000]}


# In[121]:


search_criteria = {'strategy': "RandomDiscrete", 
                   'max_models': 50,
                   'max_runtime_secs' : 900,
                   'stopping_tolerance':0.001,
                   'stopping_rounds':15,
                   'seed': 1234}


# In[122]:


# Set up DNN grid search
# Add a seed for reproducibility
dnn_rand_grid = H2OGridSearch(
                    H2ODeepLearningEstimator(
                        model_id = 'dnn_rand_grid', 
                        seed = 1234,
                        fold_assignment = "Stratified",   
                        score_validation_samples=0,
                        nfolds=3,
                        keep_cross_validation_predictions = True), 
                        search_criteria = search_criteria, 
                        hyper_params = hyper_params)


# In[112]:


# Use .train() to start the grid search
dnn_rand_grid.train(x = features, 
                    y = 'y', 
                    validation_frame=fb_valid,
                    training_frame = fb_train)


# In[113]:


#Compare model performance
dnn_best_models = dnn_rand_grid.get_grid(sort_by='MSE')
dnn_best_models


# In[114]:


# Grab the top DNN model, chosen by validation MSE
dnn_best_model = dnn_best_models.models[0]
dnn_best_model.scoring_history()
dnn_best_model


# In[115]:


dnn_best_models.get_hyperparams_dict('Grid_DeepLearning_py_22_sid_9e4b_model_python_1555300895111_1_model_33', display=True)


# In[116]:


dnn_best_model.score_history()


# In[117]:


# get the scoring history for the best model
scoring_history = pd.DataFrame(dnn_best_model.score_history())

# plot the validation and training logloss
scoring_history.plot(x='epochs', y = ['training_rmse', 'validation_rmse'])


# In[118]:


act_valid_acc_values=[]
act_train_acc_values=[]


for i,method in enumerate(["Tanh","Maxout","Rectifier"]):
    act_model=H2ODeepLearningEstimator(
                       model_id = 'dnn_rand_grid', 
                       seed = 1234,
                       nfolds=3,
                       fold_assignment='Stratified',
                       keep_cross_validation_predictions = True,activation=method,epochs = 50,epsilon=0.0001,
                       hidden=[],input_dropout_ratio=0.0,l1=0.0,l2=0.0,max_w2=100.0,
                       momentum_stable=0.5,momentum_start=0.0,rate=0.0,rate_annealing=1e-07,rho=0.999)
    
    act_model.train(x = features, y = 'y', training_frame = fb_train,validation_frame=fb_valid)
    
    act_train_acc_values.append(1-act_model.model_performance()['MSE'])
    
    act_valid_acc_values.append(1-act_model.model_performance(valid=True)['MSE'])
    
    
    


# In[123]:


plt.plot(["Maxout","Tanh","RectifierWithDropout"], act_train_acc_values)
plt.plot(["Maxout","Tanh","RectifierWithDropout"], act_valid_acc_values)


plt.xlabel('Activation')
plt.ylabel('Accuracy')
plt.title('Activation functions Vs Accuracy')

plt.legend(['Train Accuracy', 'Validation Accuracy'])


# In[125]:


valid_acc_values=[]
train_acc_values=[]

for i in range(45,55):
    if(i==45):
        epoch_model = H2ODeepLearningEstimator(
                       model_id = 'dnn_rand_grid', 
                       seed = 1234,
                       nfolds=3,
                       fold_assignment='Stratified',
                       keep_cross_validation_predictions = True,activation='tanh',epochs = i,epsilon=1.0E-10,
                       hidden=[3],input_dropout_ratio=0.1,l1=0,l2=0.0,max_w2=10.0,
                       momentum_stable=0.0,momentum_start=0.0,rate=0.005,rate_annealing=1e-06,rho=0.999) 
    else:
        epoch_model = H2ODeepLearningEstimator(
                       model_id = 'dnn_rand_grid', 
                       checkpoint=epoch_model,
                       seed = 1234,
                       nfolds=3,
                       fold_assignment='Stratified',
                       keep_cross_validation_predictions = True,activation='Maxout',epochs = i,epsilon=1.0E-6,
                       hidden=[16],input_dropout_ratio=0.1,l1=0,l2=0.0,max_w2=1000.0,
                       momentum_stable=0.0,momentum_start=0.5,rate=0.01,rate_annealing=1e-08,rho=0.999)
        

       
    epoch_model.train(x = features, y = 'y', training_frame = fb_train,validation_frame=fb_valid)
    
    
    
    train_acc_values.append(1-epoch_model.model_performance()['MSE'])
    valid_acc_values.append(1-epoch_model.model_performance(valid=True)['MSE'])
    
    


# In[126]:


plt.plot(range(45,55), train_acc_values)
plt.plot(range(45,55), valid_acc_values)


plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend(['Train Accuracy', 'Validation Accuracy'])


# In[128]:


hid_valid_acc_values=[]
hid_train_acc_values=[]


for i,method in enumerate([[3],[2],[3,3],[2,2],[3,3,3]]):
    hid_model=H2ODeepLearningEstimator(
                       model_id = 'dnn_rand_grid', 
                       seed = 1234,
                       nfolds=5,
                       fold_assignment='Stratified',
                       keep_cross_validation_predictions = True,activation='tanh',epochs = 45,epsilon=1.0E-10,
                       hidden=method,input_dropout_ratio=0.0,l1=0,l2=0.0,max_w2=10.0,
                       momentum_stable=0.0,momentum_start=0.0,rate=0.005,rate_annealing=1e-06,rho=0.999)
    
    hid_model.train(x = features, y = 'y', training_frame = fb_train,validation_frame=fb_valid)
    
    hid_train_acc_values.append(1-hid_model.model_performance()['MSE'])
    hid_valid_acc_values.append(1-hid_model.model_performance(valid=True)['MSE'])
    
    


# In[131]:


plt.plot(['[3]','[2]','[3,3]','[2,2]','[3,3,3]'], hid_train_acc_values)
plt.plot(['[3]','[2]','[3,3]','[2,2]','[3,3,3]'], hid_valid_acc_values)


plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.title('Hidden Layer Vs Accuracy')

plt.legend(['Train Accuracy', 'Validation Accuracy'])


# In[155]:


final_model=H2ODeepLearningEstimator(
                       model_id = 'dnn_rand_grid', 
                       seed = 1234,
                       nfolds=3,
                       fold_assignment='Stratified',
                       keep_cross_validation_predictions = True,activation='tanh',epochs = 45, epsilon=1.0E-10,
                       hidden=[2,2],input_dropout_ratio=0.0,l1=0,l2=0.0,max_w2=10.0,
                       momentum_stable=0.0,momentum_start=0.0,rate=0.005,rate_annealing=1e-06,rho=0.999)
final_model.train(x = features, y = 'y', training_frame = fb_train,validation_frame=fb_test)


# In[156]:


final_model.confusion_matrix()


# In[159]:


perf = final_model.model_performance()  
perf.plot()


# In[160]:


#############KNN###################
# split into input (X) and output (Y) variables
X = selectfeatures.drop(['y'],axis=1)
Y = selectfeatures['y']


# In[161]:


xtrain,x_test,ytrain,y_test=train_test_split(X,Y,test_size=0.3,random_state=2)
x_train,x_val,y_train,y_val=train_test_split(xtrain,ytrain,test_size=0.3,random_state=2)


# In[164]:


# optimum k
accuracy_val = [None]*10
count = 0
for k in range(5,15):
        KNN = KNeighborsClassifier(n_neighbors=k)
        KNN.fit(x_train,y_train)
        print(k, KNN.score(x_val,y_val))
        accuracy_val[count] = KNN.score(x_val,y_val)
        count+=1


# In[165]:


plt.ylim([.85,.90])
plt.plot(range(5,15),accuracy_val)
plt.title("Val_Accuracy vs K values")
plt.xlabel("K values")
plt.ylabel("Accuracy on Validation Set")


# In[167]:


from sklearn.neighbors import KNeighborsClassifier
metric = ["euclidean","manhattan","hamming"]
accuracy = [None]*3
count=0
for p in range(0,3):
    KNN = KNeighborsClassifier(n_neighbors = 12, metric = metric[p])
    KNN.fit(x_train, y_train)
    accuracy[count] = KNN.score(x_val,y_val)
    count+=1


# In[170]:


plt.ylim([0.85,0.9])
plt.plot(metric,accuracy)
plt.title("Accuracy vs Metric function")
plt.xlabel("Metric function")
plt.ylabel("Model Accuracy")
print(accuracy)


# In[171]:


KNN = KNeighborsClassifier(n_neighbors = 12, metric = metric[1])
KNN.fit(x_train, y_train)
print("Training Score:"+ str(KNN.score(x_train,y_train)))
print("Test Accuracy Score:"+ str(KNN.score(x_test,y_test)))
predictions_knn = KNN.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Confusion Matrix:")          
print(confusion_matrix(y_test,predictions_knn))


# In[172]:


fpr, tpr, threshold = metrics.roc_curve(y_test, predictions_knn)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("ROC_AUC Score : " + str(roc_auc) )

