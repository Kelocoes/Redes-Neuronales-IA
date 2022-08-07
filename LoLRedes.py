import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## lectura de los datos
datos = pd.read_csv('games.csv', sep=';')

cLargo = len(datos) 
cEntrenamiento = int(cLargo*0.8) # 80% para entrenar y 20% para probar
cPruebas = cLargo - cEntrenamiento
#print(cLargo, cEntrenamiento, cPruebas)

c_entrenamiento, c_pruebas = train_test_split(datos, train_size = cEntrenamiento,  test_size = cPruebas)

#print(c_entrenamiento.shape)

#print(c_entrenamiento.head())

#print(c_entrenamiento.info())

#print(c_entrenamiento.describe())

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#Game Duration
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

x_entrenamiento = full_pipeline.fit_transform(c_entrenamiento)

#print(x_entrenamiento.shape)
#print(x_entrenamiento[0,:])
#print(x_entrenamiento[1,:])
#print(x_entrenamiento[2,:])

y_entrenamiento = c_entrenamiento["winner"]
#print(y_entrenamiento)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings(action= 'ignore')

x_pruebas = full_pipeline.transform(c_pruebas)
y_test = c_pruebas["winner"]


#MODELO 1
modelo_lol1 = MLPClassifier(activation='logistic',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,8), random_state=123)
modelo_lol1.fit(x_entrenamiento,y_entrenamiento)
scores1 = cross_val_score(modelo_lol1, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores1)
print(scores1.mean())

x_pruebas = full_pipeline.transform(c_pruebas)
y_pred1 = modelo_lol1.predict(x_pruebas)
#print(y_pred1)


print("Modelo 1", accuracy_score(y_test, y_pred1))


confusion_matrix1 = confusion_matrix(y_test, y_pred1)
print(confusion_matrix1)


ax = plt.subplot()
sns.heatmap(confusion_matrix1, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
#plt.show()
"""
#--------------------------------MODELO 2-------------------------------#
modelo_lol2 = MLPClassifier(activation='tanh',solver='adam', alpha=1e-5,hidden_layer_sizes=(5,20,1,10), random_state=123)
modelo_lol2.fit(x_entrenamiento,y_entrenamiento)
scores2 = cross_val_score(modelo_lol2, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores2)
print(scores2.mean())

y_pred2 = modelo_lol2.predict(x_pruebas)
#print(y_pred2)


print("Modelo 2", accuracy_score(y_test, y_pred2))


confusion_matrix2 = confusion_matrix(y_test, y_pred2)
print(confusion_matrix2)


ax = plt.subplot()
sns.heatmap(confusion_matrix2, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
plt.show()

#--------------------------------MODELO 3-------------------------------#
modelo_lol3 = MLPClassifier(activation='tanh',solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 5, 1, 12), random_state=123)
modelo_lol3.fit(x_entrenamiento,y_entrenamiento)
scores3 = cross_val_score(modelo_lol3, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores3)
print(scores3.mean())

y_pred3 = modelo_lol3.predict(x_pruebas)
#print(y_pred3)

print("Modelo 3", accuracy_score(y_test, y_pred3))

confusion_matrix3 = confusion_matrix(y_test, y_pred3)
print(confusion_matrix3)

ax = plt.subplot()
sns.heatmap(confusion_matrix3, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
plt.show()

#--------------------------------MODELO 4-------------------------------#
modelo_lol4 = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(30, 30, 30), random_state=123)
modelo_lol4.fit(x_entrenamiento,y_entrenamiento)
scores4 = cross_val_score(modelo_lol4, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores4)
print(scores4.mean())

y_pred4 = modelo_lol4.predict(x_pruebas)
#print(y_pred4)

print("Modelo 4", accuracy_score(y_test, y_pred4))

confusion_matrix4 = confusion_matrix(y_test, y_pred4)
print(confusion_matrix4)

ax = plt.subplot()
sns.heatmap(confusion_matrix4, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
plt.show()

"""
#--------------------------------MODELO 5-------------------------------#
modelo_lol5 = MLPClassifier(activation='tanh',solver='adam', alpha=1e-5,hidden_layer_sizes=(10,1,20,1,30), random_state=123)
modelo_lol5.fit(x_entrenamiento,y_entrenamiento)
scores5 = cross_val_score(modelo_lol5, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores5)
print(scores5.mean())

y_pred5 = modelo_lol5.predict(x_pruebas)
#print(y_pred5)


print("Modelo 5",accuracy_score(y_test, y_pred5))


confusion_matrix5 = confusion_matrix(y_test, y_pred5)
print(confusion_matrix5)


ax = plt.subplot()
sns.heatmap(confusion_matrix5, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
#plt.show()

"""
#--------------------------------MODELO 6-------------------------------#
modelo_lol6 = MLPClassifier(activation='tanh',solver='adam', alpha=1e-5,hidden_layer_sizes=(10,1,20,1,30), random_state=123, beta_1 = 0.5)
modelo_lol6.fit(x_entrenamiento,y_entrenamiento)
scores6 = cross_val_score(modelo_lol6, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores6)
print(scores6.mean())

y_pred6 = modelo_lol6.predict(x_pruebas)
#print(y_pred6)


print("Modelo 6",accuracy_score(y_test, y_pred6))

confusion_matrix6 = confusion_matrix(y_test, y_pred6)
print(confusion_matrix6)


ax = plt.subplot()
sns.heatmap(confusion_matrix6, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
#plt.show()

#--------------------------------MODELO 7-------------------------------#
modelo_lol7 = MLPClassifier(activation='tanh',solver='adam', alpha=1e-5,hidden_layer_sizes=(10,1,20,1,30), random_state=123, beta_1 = 0)
modelo_lol7.fit(x_entrenamiento,y_entrenamiento)
scores7 = cross_val_score(modelo_lol7, x_entrenamiento, y_entrenamiento, cv = 10, scoring = 'accuracy')
print(scores7)
print(scores7.mean())

y_pred7 = modelo_lol7.predict(x_pruebas)
#print(y_pred7)


print("Modelo 7",accuracy_score(y_test, y_pred7))


confusion_matrix7 = confusion_matrix(y_test, y_pred7)
print(confusion_matrix7)


ax = plt.subplot()
sns.heatmap(confusion_matrix7, annot=True, ax = ax, fmt='d')

ax.set_xlabel('Etiquetas Estimadas')
ax.set_ylabel('Etiquetas reales') 
ax.set_title('Matriz de confusión - Ganadores del LoL') 
ax.xaxis.set_ticklabels(['Negative=0', 'Positive=1'])
ax.yaxis.set_ticklabels(['Negative=0', 'Positive=1'])
#plt.show()
"""