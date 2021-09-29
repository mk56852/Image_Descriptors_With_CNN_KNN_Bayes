#!/usr/bin/env python
# coding: utf-8

# In[1]:


# les bibliothéques 
import numpy as np 
import cv2 


# In[2]:


# Retourner la valeur du pixel de coordonnées (x,y)

def get_pixel(img,x,y) :
    if ((x < 0) or (y<0) or (x>=img.shape[0]) or (y>=img.shape[1]) ) :
        return False 
    else : 
        return img[x][y]
    
    
    
    
    
    
    
    
#determiner les coordonnées du point de depart de l'algorithme de pavlidis 

def pt_depart(img) :
    x = img.shape
    for i in range((x[0]-1),-1,-1) :
        for j in range((x[1]-1),-1,-1) :
            if(img[i][j]>0) :
                return [i,j]
            
            
            


    
            

#C'est l'un des algorithmes de traçage de contour les plus récents, il suit les points de contour et les extrait   
# l'extraction des points du contour :
# cette fonction va retourner une liste qui contient les coordonées de chaque point du contour 
# sous forme d'un nombre complexe : x + iy


#    Reference de direction:

#                0
#                |
#                |
#         1 <- - | - -> 3
#                |
#                |
#                2
#
   

def pavlidis(img) :
    
    
    pt_dep = pt_depart(img)       #point de depart
    pt_act = pt_dep               # point actuel 
    direction = 1               
    nb_rotation = 0              # nombre de rotation         / nb_rotation = 4 -> point isolé
    
    contour = list()              
    contour.append(pt_act)
    
    
    
    while True:
        if direction == 0: #vers le haut
            
            if get_pixel(img,pt_act[0] - 1 ,pt_act[1]-1):
                direction = 1
                pt_act = [pt_act[0] - 1 , pt_act[1]-1 ]
                contour.append(pt_act)
                nb_rotation = 0
             
            elif get_pixel(img, pt_act[0] - 1, pt_act[1] ):
                pt_act =  [pt_act[0] - 1 , pt_act[1] ]
                contour.append(pt_act)
                nb_rotation = 0
            
            
            elif get_pixel(img, pt_act[0]-1, pt_act[1]+1):
                pt_act =  [pt_act[0] - 1 , pt_act[1]+1 ]
                contour.append(pt_act)
                nb_rotation = 0
                   
             
            else:
                direction = 3
                nb_rotation += 1
                
                 
                if nb_rotation == 4:
                    break
                continue
        
        
        elif direction == 1: # vers le gauche
            if get_pixel(img,pt_act[0]+1, pt_act[1]-1):
                direction = 2
                pt_act =  [pt_act[0] + 1 , pt_act[1]-1 ]
                contour.append(pt_act)
                nb_rotation = 0
                
             
            elif get_pixel(img,pt_act[0], pt_act[1]-1):
                pt_act =  [pt_act[0]  , pt_act[1]-1 ]
                contour.append(pt_act)
                nb_rotation = 0

                
            elif get_pixel(img,pt_act[0]-1, pt_act[1]-1):
                pt_act =  [pt_act[0] -1 , pt_act[1]-1 ]
                contour.append(pt_act)
                nb_rotation = 0
               
             
            else:
                direction = 0
                nb_rotation += 1
                if nb_rotation == 4:
                    break
                continue
        elif direction == 2: # +x
            if get_pixel(img,pt_act[0]+1, pt_act[1]+1):
                direction = 3
                pt_act =  [pt_act[0]+1 , pt_act[1]+1 ]
                contour.append(pt_act)
                nb_rotation = 0
            
            
            elif get_pixel(img,pt_act[0]+1, pt_act[1]):
                pt_act =  [pt_act[0]+1 , pt_act[1] ]
                contour.append(pt_act)
                nb_rotation = 0
                
                
            elif get_pixel(img,pt_act[0]+1, pt_act[1]-1):
                pt_act =  [pt_act[0]+1 , pt_act[1]-1]
                contour.append(pt_act)
                nb_rotation = 0
                
            
            else:
                direction = 1
                nb_rotation += 1
                if nb_rotation == 4:
                    break
                continue
        else:  #direction = 3 -> droite
            
            if get_pixel(img,pt_act[0]-1, pt_act[1]+1):
                direction = 0
                pt_act =  [pt_act[0]-1 , pt_act[1]+1]
                contour.append(pt_act)
                nb_rotation = 0
              
                
            elif  get_pixel(img,pt_act[0], pt_act[1]+1):
                pt_act =  [pt_act[0] , pt_act[1]+1]
                contour.append(pt_act)
                nb_rotation = 0
                
                
            elif  get_pixel(img,pt_act[0]+1, pt_act[1]+1):
                pt_act =  [pt_act[0]+1 , pt_act[1]+1]
                contour.append(pt_act)
                nb_rotation = 0
               
            
            else:
                direction = 2
                nb_rotation += 1
                if nb_rotation == 4:
                    break
                continue
        
        if (pt_act == pt_dep) :
            break
    contour.pop()
    contour  = np.array(contour)
    
    
    contour_complexe = np.empty(contour.shape[:-1], dtype=complex)
    contour_complexe.real = contour[:, 1]
    contour_complexe.imag = contour[:, 0]
    
    return contour_complexe

        
 


def getXY(contour) :
    x = np.array(contour.real , dtype = int ) 
    y = np.array(contour.imag , dtype = int ) 
    return x , y 

