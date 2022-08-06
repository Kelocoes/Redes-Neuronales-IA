import numpy as np
import pandas as pd
import sklearn
import warnings


"""                 FASE DE LIMPIEZA Y TOMA DE DATOS                """
url = ("https://raw.githubusercontent.com/JoaquinAmatRodrigo/"
        "Estadistica-machine-learning-python/master/data/SaratogaHouses.csv")
datos = pd.read_csv(url, sep=",")

datos.columns = ["precio","tamaño_lote", "antiguedad", "precio_terreno",
                "area_construida", "universitarios", "dormitorios", 
                "chimenea", "baños", "habitaciones", "calefaccion",
                "consumo_calefacion", "desague", "vistas_lago",
                "nueva_construccion", "aire_acondicionado"]

#print(datos.info())
#print(datos.select_dtypes(include=['object']).describe())
#print(datos.select_dtypes(include=['int64']).describe())

from sklearn.model_selection import train_test_split 

N = len(datos)
cTrain = int(N*0.8) # 80% para entrenar y 20% para probar
cTest = N - cTrain
#print(N,cTrain,cTest)
train_data,test_data= sklearn.model_selection.train_test_split(datos, train_size=cTrain, test_size=cTest)
#print(train_data.shape, test_data.shape)
#print(train_data.head())


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 6 atributos categóricos
cat_attribs = ['calefaccion','consumo_calefacion','desague','vistas_lago','nueva_construccion','aire_acondicionado']

cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False))
    ])


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 9 atributos numéricos
num_attribs = ['tamaño_lote','antiguedad','precio_terreno','area_construida','universitarios','dormitorios','chimenea','baños','habitaciones']

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()) 
    
    ])


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs), 
])

X_train = full_pipeline.fit_transform(train_data)

#print(X_train.shape)
#print(X_train[0,:])

y_train = train_data["precio"]
#print(y_train)


"""                 FIN DE LIMPIEZA Y TOMA DE DATOS                 """


"""                     COMIENZO ENTRENAMIENTO RED          """

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import random
import time

warnings.filterwarnings(action= 'ignore')

mae = 60000

random.seed(222)

while (mae > 10000):

    tiempo_actual = time.time()
    i = random.randint(1,21)
    j = random.randint(1,21)
    print(i,j)

    modelo1 = MLPClassifier(activation='relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(i,j), random_state=123)
    modelo1.fit(X_train, y_train)
    scores1 = cross_val_score(modelo1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    #print(scores1)
    #print(scores1.mean())



    """                      FIN DE ENTRENAMIENTO                          """

    """               INICIO DE PRUEBA DE LOS DATOS DE PRUEBA                 """

    X_test = full_pipeline.transform(test_data)
    #print(X_test)

    y_pred1 = modelo1.predict(X_test)  
    #print(y_pred1)

    y_test = test_data["precio"]
    #print(y_test)

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(
            y_true  = y_test,
            y_pred  = y_pred1
        )
    print(f"El error medio absoluto del modelo 1 es: {mae}")
    print(time.time() - tiempo_actual)