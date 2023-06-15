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

##############################################################################
#################### IMPLÉMENTATION DE L'ARBRE DE DÉCISION ###################
##############################################################################

############################# CLASSE NOEUD  ##################################

#Objectif classe noeud :  Représenter chaque noeud dans l'arbre de decision (noeud = 
#structure représentant une condition de division sur un attribut ou une prédiction 
#finale)

class noeud:
    
    def __init__(self, ind_attr=None, seuil=None, gauche=None, droite=None, val=None, effectif=None,proportion=None):
        '''
        Initialisation des paramètres nécessaires pour représenter 
        un nœud de l'arbre de décision, y compris les informations sur l'attribut 
        de division (index de l'attribut pour lequel la division est effectué), 
        le seuil (valeur seuil pour la division), les sous-arbres gauche (valeurs 
        <= seuil) et droit (valeurs > seuil), la valeur prédite (par le noeud (pour 
        les feuilles uniquement)), l'effectif et la proportion. 
        '''
        self.ind_attr = ind_attr  
        self.seuil = seuil  
        self.gauche = gauche  
        self.droite = droite  
        self.val = val  
        self.effectif = effectif 
        self.proportion = proportion 
  
    def est_feuille(self):
            
        '''
        Vérifier si un nœud donné est une feuille de l'arbre, ie s'il n'a pas de fils.
        '''
        return self.gauche is None and self.droite is None

####################### CLASSE ARBRE DE DÉCISION #############################

#Objectif classe arbre_decision : représenter l'arbre de décision et de fournir les fonctionnalités
#pour sa construction, son entraînement et son utilisation.

class arbre_decision:
    
    ##Étape 1 : Initialisation des attributs de l'objet de la classe. 
    def __init__(self, min_ech_split=2, prof_max=2, critere='gini'):
        '''    
        Les attributs initialisés dans le constructeur sont :
        - le nombre minimum d'échantillons requis pour diviser un noeud de l'arbre.
        - la profondeur maximale de l'arbre.
        - la racine de l'arbre.
        - le critère d'impureté utilisé pour la division des noeuds de l'arbre.
        - le nom des variables explicatives.

        Par défaut, le critère est initialisé à "gini".
        '''
        self.min_ech_split = min_ech_split  
        self.prof_max = prof_max  
        self.racine = None
        self.critere = critere  
        self.noms_variables = None 

    ##Étape 2 : Création des méthodes pour le calcul des critères d'impureté
    def calcul_entropy(self, y):
        '''
        Fonction qui calcule l'entropie (mesure la quantité d'informations d'une 
        variable ou d'un événement), on calcule d'abord la probabilité de chaque étiquette, 
        on supprime lorsque une étiquette n'apparait pas puis on calcule l'entropie.
        '''
        prob = np.bincount(y) / len(y)  
        prob = prob[prob != 0]  #
        entropy = -np.sum(prob * np.log2(prob))  
        return entropy
    
    def calcul_gini(self, y):
        '''
        Fonction qui calcule l'impurete de Gini, similaire à l'entropie sauf que
        Gini n'utilise pas de logarithme.
        '''
        prob = np.bincount(y) / len(y)   
        prob = prob[prob != 0] 
        gini = 1 - np.sum(prob**2)  # calcule de l'impurete de Gini ()
        return gini

    def calcul_chi2(self, y):
        '''
        Fonction qui calcule le Chi² (mesure statistique pour tester l'hypothèse
        nulle de l'indépendance des variables.). Pour le chi2, on prend en compte 
        le nombre de chaque étiquette unique (stocké dans "nb"). 
        Si la moyenne de "nb" est égal à 0, chi2 = 0 car le dénominateur ne peut pas 
        être égale à 0 dans une division.
        '''
        nb = np.bincount(y)  
        if nb.mean() != 0:
            chi2 = np.sum((nb - nb.mean()) ** 2 / nb.mean())
        else:
            chi2 = 0 
        return chi2  

    def calcul_impurete(self, y):
        '''
        Fonction qui permet de définir la mesure d'impureté que l'on veut utiliser.
        Si le critère donné ne correspond à aucune de ces options, la fonction affiche 
        un message d'erreur et donne une liste des critères valides.
        '''
        if self.critere == 'gini':
            return self.calcul_gini(y)
        elif self.critere == 'entropy':
            return self.calcul_entropy(y)
        elif self.critere == 'chi2':
            return self.calcul_chi2(y)
        else:
            print(f"Critère d'impureté non valide : {self.critere}") 
            print("Liste des critères valides : -gini \n -entropy \n chi2")

    ##Étape 3 : Trouver le meilleur seuil pour diviser un attribut en minimisant
    #l'impureté choisi. 
    def meilleur_seuil(self, X, y):
        '''   
        La fonction itère sur chaque attribut et utilise sa valeur médiane comme seuil 
        pour séparer les données.
        '''
        meilleur_indice = None
        meilleur_seuil = None
        meilleur_score = float('inf')
       
        for indice in range(X.shape[1]):
            X_col = X[:, indice]  
            seuil = np.median(X_col)  
            separation = X_col <= seuil
            y_g, y_d = y[separation], y[~separation]
            score = (len(y_g)*len(y)) * self.calcul_impurete(y_g) + (len(y_d)*len(y)) * self.calcul_impurete(y_d)
           
            if score < meilleur_score:
                meilleur_indice = indice
                meilleur_seuil = seuil
                meilleur_score = score
       
        return meilleur_indice, meilleur_seuil
   
    ##Étape 4 : Création d'une feuille de l'arbre. 
    def calcul_val_feuille(self, y):
        '''
        La fonction permet de trouver la valeur (modalité de la variable cible) qui 
        apparait le plus sur une feuille (en comptant le nombre de chaque étiquette 
        unique). S'il n'y a aucun échantillon, elle retourne 0, sinon retourne l'etiquette 
        majoritaire et l'effectif, celle-ci sera utilisée pour les prédictions. 
        '''
        val, nb = np.unique(y, return_counts=True)  
        if len(nb) == 0:
            return 0, len(y), None
        else:
            return val[np.argmax(nb)],len(y) 

    ##Étape 5 : Construction de l'arbre
    def construct(self, X, y, prof, parent_effectif=None):
        '''
        Fonction qui construit l'arbre de decision en utilisant un algorithme récursif 
        et les fonctions que nous avons défini ci-dessus. 
        Quand on atteint les conditions d'arret (profondeur maximale, nombre 
        minimum d'échantillons, ou un seul label), on retourne une feuille.
        Sinon elle divise les données sur le meilleur attribut et le meilleur seuil et 
        continue à construire les sous-arbres gauche et droit.
        '''
        if parent_effectif is None:
            parent_effectif = len(y)  

        if prof >= self.prof_max or len(np.unique(y)) == 1 or X.shape[0] < self.min_ech_split:
            val_feuille, effectif = self.calcul_val_feuille(y)  
            proportion = effectif / parent_effectif  
            return noeud(val=val_feuille, effectif=effectif, proportion=proportion)  
        
        ind_attr, seuil = self.meilleur_seuil(X, y)  
        separation = X[:, ind_attr] <= seuil  
        effectif = len(y)  
        gauche = self.construct(X[separation], y[separation], prof + 1, parent_effectif=effectif)  
        droite = self.construct(X[~separation], y[~separation], prof + 1, parent_effectif=effectif) 
        return noeud(ind_attr, seuil, gauche, droite, effectif=effectif, proportion=effectif / parent_effectif)

    ##Étape 6 : Entraînement de l'arbre de décision. On construit l'arbre de decision 
    #à partir de la racine.
    def train(self, X, y):
        self.racine = self.construct(X, y, 0)  
     
    ##Étape 7 : Affichage de l'arbre de décision dans la console.
    def affiche_noeud(self, noeud, noms_variables, indent=''):
        ''' 
        CAffiche de manière récursive chaque noeud indiquant la condition de
        division pour les noeuds internes et la prédiction / l'effectif / la proportion 
        pour les feuilles.
        '''
        if noeud.est_feuille():
            print(indent + "Prédiction :", noeud.val)
            print(indent + "Effectif :", noeud.effectif)
            print(indent + "Proportion :", round(noeud.proportion*100,1),"%")        
        else:
            nom_variable = None
            if noms_variables is not None:
                nom_variable = noms_variables[noeud.ind_attr]
            else:
                nom_variable = f"Attribut {noeud.ind_attr}"
            print(f"{indent}\033[1;34m{nom_variable} <= {noeud.seuil}\033[0m")  
            print(f"{indent}\033[1;32m--> Vrai :\033[0m")  
            self.affiche_noeud(noeud.gauche, noms_variables, indent + "   ")
            print(f"{indent}\033[1;31m--> Faux :\033[0m") 
            self.affiche_noeud(noeud.droite, noms_variables, indent + "   ")
     
    def affiche_arbre(self, custom_noms_variables=None):
        ''' 
        Affiche l'arbre à partir de la racine utilisant des noms de variables 
        personnalisés si fournis.
        '''
        noms_variables = custom_noms_variables or self.noms_variables 
        self.affiche_noeud(self.racine, noms_variables)

    ##Étape 8 : Prédictions sur l'ensemble de test
    def _predict(self, echantillon, noeud):
        '''
        Fonction qui permet d'effectuer une prédiction pour un échantillon unique. 
        Si le noeud est une feuille, elle retourne sa valeur. Si la valeur de l'attribut 
        est inferieure au seuil, elle parcourt le sous-arbre gauche. Sinon parcourt 
        le sous arbre droit.
        '''
        if noeud.est_feuille():  
            return noeud.val
        if echantillon[noeud.ind_attr] < noeud.seuil: 
            return self._predict(echantillon, noeud.gauche)
        else:  
            return self._predict(echantillon, noeud.droite)

    def predict(self, X):
        ''' 
        Applique la fonction _predict à chaque echantillon.
        '''
        return np.array([self._predict(echantillon, self.racine) for echantillon in X])  










