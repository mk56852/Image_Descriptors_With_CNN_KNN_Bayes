#!/usr/bin/env python
# coding: utf-8

# In[4]:


import data as d
import classifieurs as cl
import numpy as np


# In[3]:


#importation des données de la base MPEG7

data_train , id_train, labels_train  =  d.getData("C:/Users/dell/Desktop/PCD/PNG_data2/" ,nb_img_par_classe=20, descripteur= 'norme', nb_pt_rep = 100 )
data_t = np.absolute(data_train)


# In[5]:


# cnn model training 

#data
X_train, X_test, y_train, y_test = d.split_data(data_t,id_train , cnn = True,test = 0.2)


#model_train
CNNmodel = cl.cnn_model(100,70)
cl.train_cnn_model(CNNmodel,X_train, y_train, X_test, y_test, batch = 100,nb_epochs=160)


# In[6]:


CNNmodel.save("CNNmodel.h5")


# In[14]:


# bayes model 



#data
X_train, X_test, y_train, y_test = d.split_data(data_t,id_train , cnn = False , test=0.2)


#model_train
BAYESmodel = cl.bayes_model()
cl.train_bayes_model(BAYESmodel,X_train,y_train)


#score
print("Bayes Model =  " + str(BAYESmodel.score(X_test, y_test)))


# In[15]:


# KNN model



#data


#model_train
KNNmodel = cl.knn_model() 
cl.train_knn_model(KNNmodel ,X_train , y_train)


#prediction
predicted= KNNmodel.predict(X_test)
print("KNNmodel = " + str(cl.score(predicted,y_test)))

