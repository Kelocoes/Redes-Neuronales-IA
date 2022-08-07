import sklearn
import numpy as np
import pandas as pd

## se carga el archivo y se separa entrenamiento y prueba
from sklearn.model_selection import train_test_split

datos = pd.read_csv('games.csv', sep = ';')

cLargo = len(datos)
cEntrenamiento = int(cLargo*0.8) ## 80% para entrenar y 20% para pruebas
cPruebas = cLargo - cEntrenamiento
## print (cLargo, cEntrenamiento, cPruebas)

c_entrenamiento, c_pruebas = train_test_split(datos, train_size= cEntrenamiento, test_size = cPruebas)

# print(c_entrenamiento.shape)
# print(c_entrenamiento.head())
# print(c_entrenamiento.info())
# print(c_entrenamiento.describe())

## pipeline de los atributos numericos:

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder ## ? 

## Game duration:
numero_atributos_num0 = ["gameDuration"]

# Atributos categoricos: 
numero_atributos_cat = ["firstBlood",
                        "firstTower",
                        "firstInhibitor",
                        "firstBaron",
                        "firstDragon",
                        "firstRiftHerald"]
# Atributos numericos: 
 
numero_atributos_num = ["t1_towerKills",
                        "t1_inhibitorKills",
                        "t1_baronKills",
                        "t1_dragonKills",
                        "t1_riftHeraldKills",
                        "t2_towerKills",
                        "t2_inhibitorKills",
                        "t2_baronKills",
                        "t2_dragonKills",
                        "t2_riftHeraldKills"]

numero_pipeline = Pipeline([
   ("imputer", SimpleImputer(strategy = "most_frequent")),
   ("scaler", StandardScaler())
])

categorico_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("scaler", OneHotEncoder(sparse = False))
])
## Pipeline Completo

from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("num0", numero_pipeline, numero_atributos_num0),
    ("cat", categorico_pipeline, numero_atributos_cat),
    ("num", numero_pipeline, numero_atributos_num)
])

## Extraemos las etiquetas de clase

x_entrenamiento = full_pipeline.fit_transform(c_entrenamiento)

# print(x_entrenamiento.shape)
# print(x_entrenamiento[0, :])
# print(x_entrenamiento[1, :])
# print(x_entrenamiento[2, :])

y_entrenamiento = c_entrenamiento["winner"]

## Analizar desempeño

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings(action= 'ignore')

x_pruebas = full_pipeline.transform(c_pruebas)
y_pruebas = c_pruebas["winner"]


## ARBOLES DE DECISION
## algoritmo decision treeclassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score


## Modelo 1 - 10 :  max_depth = 100 ... 1000
x_pruebas = full_pipeline.transform(c_pruebas)

for i in range(1, 11):

    prof = 100 * i
    #print(prof)
    modeloN = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=prof, splitter = "best", random_state= 123)
    modeloN.fit(x_entrenamiento, y_entrenamiento)
    scoresN = cross_val_score(modeloN, x_entrenamiento, y_entrenamiento, cv = 5, scoring = 'accuracy')
    #print(scoresN)
    #print(scoresN.mean())

    tree.plot_tree(modeloN)
    tree.export_graphviz(decision_tree = modeloN, class_names= True, out_file = "Arbol modelo " + str(i) + ".dot")


    y_pred = modeloN.predict(x_pruebas)
    print("Modelo " + str(i) , accuracy_score(y_pruebas, y_pred))



