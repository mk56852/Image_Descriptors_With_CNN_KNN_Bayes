#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np 
import cv2
import os
import descripteurs as des
import reparametrage_euc as rep
import suivi as s


# In[13]:


def getData(src_path ,nb_img_par_classe = 20 ,  descripteur = "norme" , nb_pt_rep = 100 , label = True ) :
    
    assert descripteur in ["norme","crimmins","crimmins2" , "courbure","hog"]
    i = 0
    labels = np.array([])
    id_ = np.array([],dtype=int)
    data = list()
    j = -1
    
    for file in os.listdir(src_path) :
        # avoir les labels 
       
        if i % nb_img_par_classe == 0 :
            j+=1
            m_elt_list = list()
            name = (os.path.splitext(file)[0]).split(sep='-')[0]              
            labels = np.append(labels,[name])
            
        i +=1
        
        
        img = cv2.imread(src_path + file ,cv2.IMREAD_GRAYSCALE)
        
        
        if(descripteur == "hog") : 
            desc = des.hog(img)
        
        else :
            #detection contour
            img_canny = cv2.Canny(img , 50,100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            img_cnt = cv2.dilate(img_canny, kernel)
            #suivi contour
            contour = s.pavlidis(img_cnt)
            x_,y_ = contour.real , contour.imag
            #reparametrage 
            X1,Y1 = rep.Reparametrage_euclidien(x_,y_,nb_pt_rep)
            #calcul des  descripteur 
            if(descripteur == "norme") : 
                desc = des.norme(X1,Y1)
            elif(descripteur == "crimmins") :
                desc = des.crimmins(X1,Y1,n0,n1)
            elif(descripteur == "crimmins_normalisé") :
                desc = des.crimmins_stable(X1,Y1,n0,n1,p,q)
            else:
                desc= des.courbure(X1,Y1)

        
        id_ = np.append(id_,[j])
        data.append(desc)
    data = np.array(data)
        
    if (label) :
        return data , id_ , labels
    else :
        return data
        


# In[ ]:


def getDataDict():
    i = 0
    data = dict()
    j = -1
    src_path = "C:/Users/dell/Desktop/PCD/PNG_data2/"
    for file in os.listdir(src_path) :
        # avoir les labels 
       
        if i % 20 == 0 :
            j+=1
            
            name = (os.path.splitext(file)[0]).split(sep='-')[0]              
            data[j] = name
            
        i +=1
    return data


# In[14]:




