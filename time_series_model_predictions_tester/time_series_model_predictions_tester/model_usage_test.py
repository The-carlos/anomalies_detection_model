from sqlalchemy import create_engine
import pandas as pd
import os
import glob
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn import preprocessing
import numpy as np


loaded_model = load_model("model_jumamex.h5")
print("Model successfully loaded!")

# Crear un objeto MinMaxScaler para estandarizar los datos en el rango [0, 1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

X = pd.read_csv("to_predict.csv")
print("DataFrame to predict:")
print(X)

# Formatear los datos de entrada para que coincidan con la forma esperada por el modelo
X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))

#Crear DF con bandas y real
predictions = np.reshape([i[0][0] for i in loaded_model.predict(X)], (-1, 1))

# Las predicciones pueden tener una forma diferente, según el modelo, así que puedes imprimir para verificar
print("Predictions:")
print(predictions[0][0])

# Crear un DataFrame que contiene los resultados de las predicciones y las bandas de confianza

# Inicialmente, creamos un DataFrame 'd' con columnas para las predicciones y bandas de confianza
zero_data = np.zeros(shape=(X.shape[0], 4))
d = pd.DataFrame(zero_data, columns=["14-day_tpv", "7-day_tpv", "1-day_tpv", "tpv_predict"])


# Llenar la columna 'tpv_predict' con las predicciones del modelo
d["tpv_predict"] = predictions  # No es necesario reshape aquí


# Invertir la escala de las predicciones para obtener los valores originales
d = min_max_scaler.inverse_transform(d)
print("Inverse transform")
d = pd.DataFrame(d, columns=["14-day_tpv", "7-day_tpv","1-day_tpv","tpv_predict"])
bandas = d.drop(columns = ["14-day_tpv", "7-day_tpv","1-day_tpv"])
# Crear un nuevo DataFrame 'bandas' para las bandas de confianza y los valores reales


# Calcular las bandas de confianza como la media más/menos una desviación estándar de las predicciones

std_dev = d["tpv_predict"].std()  # Calcular la desviación estándar
bandas["lim_inf"] = d["tpv_predict"] - std_dev
bandas["lim_sup"] = d["tpv_predict"] + std_dev



# Agregar los valores reales 'tpv_real' del conjunto de prueba
bandas["tpv_real"] = y.values

# Determinar si los valores reales están dentro de las bandas de confianza y crear la columna 'In'
bandas["In"] = bandas.apply(lambda row: True if (row.tpv_real >= row.lim_inf and row.tpv_real <= row.lim_sup) else False, axis=1)

# Establecer el índice del DataFrame 'bandas' para que coincida con las fechas del conjunto de prueba
bandas = bandas.set_index(X.index)

# Mostrar el DataFrame
print("Resultados finales")
print(bandas)