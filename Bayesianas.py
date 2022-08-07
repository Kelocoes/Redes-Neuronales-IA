import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import


datos = pd.read_csv('games.csv', sep=';')
C=len(datos)
c_entrenamiento=int(C*0.8) # 80% para entrenar y 20% para probar
c_prueba=C-c_entrenamiento
#print(C,c_entrenamiento,c_prueba)
datos_entrenamiento,datos_prueba= sklearn.model_selection.train_test_split(datos, train_size=c_entrenamiento, test_size=c_prueba)

#print(datos_entrenamiento.shape)
#print(datos_entrenamiento.head)
#print(datos_entrenamiento.info())
#print(datos_entrenamiento.describe())

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

numero_atributos_num0 = ["gameDuration"]

#Atributos categoricos
numero_atributos_cat = ["firstBlood", 
                        "firstTower",
                        "firstInhibitor", 
                        "firstBaron", 
                        "firstDragon", 
                        "firstRiftHerald"]

#Atributos numericos
numero_atributos_num = [
                    "t1_towerKills" , 
                    "t1_inhibitorKills" , 
                    "t1_baronKills",	
                    "t1_dragonKills", 
                    "t1_riftHeraldKills",	
                    "t2_towerKills", 
                    "t2_inhibitorKills", 
                    "t2_baronKills", 
                    "t2_dragonKills", 
                    "t2_riftHeraldKills"]

numero_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

categorico_pipeline =   Pipeline([
    ("imputer",SimpleImputer(strategy = "most_frequent")),
    ("scaler", OneHotEncoder(sparse=False))
])


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer ([
    ("num0", numero_pipeline, numero_atributos_num0),
    ("cat", categorico_pipeline, numero_atributos_cat),
    ("num", numero_pipeline, numero_atributos_num)
])

y_entrenamiento = datos_entrenamiento["winner"]

#print(y_entrenamiento)

x_entrenamiento = full_pipeline.fit_transform(datos_entrenamiento)

#print(x_entrenamiento.shape)

#print(x_entrenamiento)


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

modelo1 = GaussianNB(var_smoothing=0.001)
modelo1.fit(x_entrenamiento, y_entrenamiento)
scores1 = cross_val_score(modelo1, x_entrenamiento, y_entrenamiento, cv=5,scoring='accuracy')
print(scores1.mean())

x_prueba = full_pipeline.transform(datos_prueba)
#print(x_prueba)

y_prediccion = modelo1.predict(x_prueba)
#print(y_prediccion)

y_prueba = datos_prueba["winner"]
#print(y_prueba)

from sklearn.metrics import accuracy_score
accuracy_score(y_prueba, y_prediccion)
print(accuracy_score(y_prueba, y_prediccion))

from sklearn.metrics import confusion_matrix

matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
print(matriz_confusion)

import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.subplot()
sns.heatmap(matriz_confusion, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
plt.show()