#!/usr/bin/env python
# coding: utf-8

# In[4]:


# l'importation du bibliothéque 

#pour Cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Conv1D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
# pour Knn
from sklearn.neighbors import KNeighborsClassifier as Knn
# pour bayes
from sklearn.naive_bayes import * 


# In[1]:


# notre 1er classifieur  -> cnn 
#input -> bands : la taille du vecteur d'entrée , num_class : le nombre de classes
 

def cnn_model(bands, num_classe):
    clf = Sequential()
    clf.add(Conv1D(20, (24), activation='relu', input_shape=(bands,1)))
    clf.add(Conv1D(20, (24), activation='relu'))
    clf.add(Conv1D(20, (24), activation='relu'))
    clf.add(Dropout(0.5))
    clf.add(Flatten())
    clf.add(Dense(100))
    clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Dense(num_classe, activation='softmax'))
    clf.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return clf


def train_cnn_model(model,x_train,y_train,x_test,y_test , batch = 100 , nb_epochs=300):
    history = model.fit(x_train, y_train, epochs = nb_epochs, batch_size = batch , validation_data=(x_test, y_test))
    return(history)




# In[6]:


# 2éme classifieur  -> Knn 



def knn_model(k = 1):
    model = Knn(n_neighbors= k )
    return model

def train_knn_model(model ,X_train , y_train) :
    model.fit(X_train,y_train)
    

# dans le model knn il faut utiliser la fonction score pour comparer les resultats predictés et les resultats attendus 

def score (predicted,y_test) :
    x = 0
    for i in range(len(predicted)) : 
        if (predicted[i] == y_test[i]) :
            x += 1
    return (x/len(predicted))


    


# In[3]:


# bayes 


def bayes_model():
    model = GaussianNB()
    return model


def train_bayes_model(model ,X_train , y_train) :
    model.fit(X_train, y_train)
    

