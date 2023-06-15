#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 31 20:16:08 2023

@author: Coraly Gervasoni, Perrine Martin & Walid Ghalleb -- M1 MIASHS LYON 2
"""

##############################################################################
############################# BIBLIOTHÈQUES ##################################
##############################################################################

#Importation des librairies nécessaires poour faire fonctionner le code
import numpy as np #pour la manipulation de tableaux et le calcul scientifique
from sklearn.datasets import load_iris  #pour importer jeu de données Iris
from sklearn.model_selection import train_test_split  #pour diviser le jeu de données en deux (entrainement / test)
from sklearn.metrics import accuracy_score  #pour calculer la précision
from sklearn.metrics import confusion_matrix #pour la matrice de confusion

#import de l'arbre de décision créé
from algo_arbre import arbre_decision

##############################################################################
#################################### TEST #################################### 
##############################################################################

######################## FIXATION DE LA GRAINE ###############################

np.random.seed(100)

############################## DONNÉES #######################################

iris = load_iris()  #chargement du jeu de données iris
X = iris.data  #on récupère les attributs (variables explicatives)
y = iris.target  #on récupère les labels (variable cible à prédire)

#on divise le jeu de données en train_set / test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

############################ ENTRAÎNEMENT ###########s#########################

#on créé et entraine l'arbre de décision en définissant le critère d'impureté sur 
#l'entropy et le maximum de feuille à 5 les autres paramètres prendront les valeurs 
#que nous avons définis par défault lors du codage de l'algorihtme
arbre = arbre_decision(prof_max=5, critere='gini')
arbre.train(X_train, y_train)
noms_variables = iris.feature_names
arbre.affiche_arbre(noms_variables)

############################# PRÉDICTIONS ####################################

#on effectue des prédictions sur l'ensemble des données de test
y_pred = arbre.predict(X_test)

################################ ACCURACY ####################################

#on calcule et affiche le score de l'accuracy. L'accuracy correspond au nombre de 
#classification correct sur le nombre total d'exactement (% de bonnes réponses).
precision = accuracy_score(y_test, y_pred)
print("\n Accuracy sur le jeu de données Iris : " + str(precision))
#L'exactitude globale du modèle est de 96.67% ce qui est très élevé. 
#Cela signifie que le modèle a correctement classé 96.67% des observations.

########################## MATRICE DE CONFUSION ##############################

#Calcul de la matrice de confusion. On compare les valeurs prédites par le modèle 
#avec les vraies valeurs. 
matrice_confusion = confusion_matrix(y_test, y_pred)
print("\n Matrice de confusion :")
print(matrice_confusion)
#11 observations de la première classe ont été correctement prédites 
#5 observations de la deuxième classe ont été correctement prédites
#1 a été incorrectement prédite comme appartenant à la troisième classe
#13 observations de la troisième classe ont été correctement prédites 







