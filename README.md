# Descripteurs_Image_avec_CNN_KNN_Bayes



notre objectif est de mener une étude comparative de quelques descripteurs et évaluer ses performance en utilisant les plus récents algorithmes d'apprentissage CNN , KNN et Bayes pour aboutir à un système intelligent basés vision capable de détecter les objets et les classifier .

le processus general est le suivant  : 

![capp](https://user-images.githubusercontent.com/65729882/135352370-19734c39-798f-4f37-b23a-71559ce205f7.PNG)




1 ) la segmentation d'image : 

on a utilisé Canny pour la detection des contours dans l'image 

Le processus générale de Canny
![canny](https://user-images.githubusercontent.com/65729882/135352789-e64dd04b-8f92-486b-8840-37d86440540d.PNG)



2) suivi de contour 

Pavlids : C'est l'un des algorithmes de traçage de contour les plus récents, il suit les points de contour et les extrait . 



3) le reparametrage euclidien


c'est la phase de la normalisation des courbes paramétrés par longueur d'arc normalisée

![rep](https://user-images.githubusercontent.com/65729882/135353123-74277301-fead-4407-87de-741c82c69e58.PNG)



4) les descripteurs 


-courbure  :    A partir d’un contours paramétrés en longueur d’arc C(t)=(x(t),y(t))     la courbure en un point est définie par :

![courbure](https://user-images.githubusercontent.com/65729882/135353568-b52395d8-b8e0-41f1-bbfe-25fc0859162c.PNG)


- les descripteurs de fourier :  Soit C(t) = x(t)+i y(t) une courbe paramétrique représentant le contour ferme d’une forme, on note par f sa transformée de Fourier discrète. 


![fourier](https://user-images.githubusercontent.com/65729882/135353586-c4ea7ae8-4aee-457b-903f-ce95f0857126.PNG)


On distingue deux ensembles de descripteurs : 


![i1i2](https://user-images.githubusercontent.com/65729882/135354135-be4c8046-2dea-437d-8756-26c5c60c9bb1.PNG)


- HOG  : Histogramme de gradient orienté



5) les classifieurs 

-CNN

-KNN

-Bayes 



6) Les Resultats Obtenus 


![KNN](https://user-images.githubusercontent.com/65729882/135353338-af1e2bec-5efd-4f47-984a-728fa8989bbc.PNG)
![bayes](https://user-images.githubusercontent.com/65729882/135353347-374fdce1-9d10-4a3e-b6b6-617080cbce1d.PNG)
![cnn](https://user-images.githubusercontent.com/65729882/135353349-c41f77c3-8079-46fc-921e-be452ca2ef38.PNG)

