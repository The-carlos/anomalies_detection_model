from sqlalchemy import create_engine
import pandas as pd
import os
import glob
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn import preprocessing
import numpy as np

def query_generator(query_input, db, output_name):

    main_address = "mysql+mysqldb://powerbi:2o2o_p0w3rbI@z32.bi.rds.reader.production.billpocket.com:3306/"

#    try:
#        f = open(output_name)
#        print("El archivo ", output_name, " ya existe.")

#    except FileNotFoundError:

    conn = create_engine(main_address + db)

    query = query_input

    df = pd.read_sql_query(query,conn)

    df.to_csv(output_name, encoding='utf-8-sig', index = False)
    print("Archivo", output_name, "creado exitosamente!")






check_models = pd.read_excel("check_model.xlsx")

def generate_queries(client_list):
    query_data = []
    base_query = """
    SELECT t.tran_fechahora AS txn_date, COALESCE(t.tran_monto,0) + COALESCE(t.tran_propina,0) AS tpv, m.client
    FROM billpocketz32.transacciones AS t
    LEFT JOIN operacion_bi.bi_merchants AS m
    ON t.tran_usuario = m.id_merchant
    WHERE t.tran_estatus = 'aprobada'
    AND (t.tran_tipotransaccion = 'venta' OR t.tran_tipotransaccion = 'devolucion')
    AND m.client IN (
    """

    for client in client_list:
        if client in ('PIG930806GL1', 'CJF941024F7A', 'OEB1410238K9', 'CDJ2011111U4', 'ANL171011549'):
            query = """
            SELECT t.tran_fechahora AS txn_date, COALESCE(t.tran_monto,0) + COALESCE(t.tran_propina,0) AS tpv, m.client
            FROM billpocketz32.transacciones AS t
            LEFT JOIN operacion_bi.bi_merchants AS m
            ON t.tran_usuario = m.id_merchant
            WHERE t.tran_estatus = 'aprobada'
            AND (t.tran_tipotransaccion = 'venta' OR t.tran_tipotransaccion = 'devolucion')
            AND m.client IN (
            'PIG930806GL1',
            'CJF941024F7A',
            'OEB1410238K9',
            'CDJ2011111U4',
            'ANL171011549'
            )
            AND t.tran_fechahora >= '2023-08-31';
            """
        elif client in ('AMT170314IP0', 'NME101203MB9'):
            query = """
            SELECT t.tran_fechahora AS txn_date, COALESCE(t.tran_monto,0) + COALESCE(t.tran_propina,0) AS tpv, m.client
            FROM billpocketz32.transacciones AS t
            LEFT JOIN operacion_bi.bi_merchants AS m
            ON t.tran_usuario = m.id_merchant
            WHERE t.tran_estatus = 'aprobada'
            AND (t.tran_tipotransaccion = 'venta' OR t.tran_tipotransaccion = 'devolucion')
            AND m.client IN (
            'AMT170314IP0',
            'NME101203MB9'
            )
            AND t.tran_fechahora >= '2023-08-31';
            """
        else:
            query = f"{base_query}'{client}') AND t.tran_fechahora >= '2023-08-31';"
        
        # Buscar el RFC en el DataFrame y agregar la consulta al DataFrame de resultados
        check_models.loc[check_models['rfc'] == client, 'query_str'] = query
    
    useful_data_models = check_models[["razon_social", "model_name", "query_str"]].copy()
    useful_data_models["model_name"] = useful_data_models["model_name"].str.replace(r'\s+', '', regex=True)
    useful_data_models = useful_data_models.drop_duplicates().reset_index(drop = True)
    useful_data_models.to_csv("check_models_w_queries.csv", index = False)
    return useful_data_models
    
def df_creator_hourly(files_names):
    path = r'C:\Users\csanchez_billpocket\Desktop\Billpocket\Data scientist\time_series_model_predictions_tester\data_for_predict\\'
    path_to_save = r'C:\Users\csanchez_billpocket\Desktop\Billpocket\Data scientist\time_series_model_predictions_tester\two_hourly_data\\'
    for file_name in files_names:
        file_df = pd.read_csv(os.path.join(path, file_name), parse_dates=['txn_date'], index_col='txn_date')

        #Verificamos que el DataFrame no esté en blanco
        if not file_df.empty:
            # Resetea el índice para evitar conflictos con el nombre 'txn_date'
            file_df.reset_index(inplace=True)
            # Crea una nueva columna 'interval' con los intervalos de 2 horas
            file_df['interval'] = file_df['txn_date'] + pd.DateOffset(hours=1)  # Suma 1 hora para redondear hacia adelante
            file_df['interval'] = file_df['interval'].dt.floor('2H')
            # Agrupa por 'interval' y suma 'tpv' Acá dejamos a un lado 'client', porque no es necesario
            test_file_data_2hourly = file_df.groupby(['interval'])['tpv'].sum().reset_index()
            # Renombramos 'interval' a 'txn_date'
            test_file_data_2hourly.rename(columns={'interval': 'txn_date'}, inplace=True)
            print(f"{file_name} successfully grouped by 2-hour intervals!")
            # Exporta el DataFrame resultante a un archivo CSV
            test_file_data_2hourly.to_csv(f"{path_to_save}_2hourly_hist_" + file_name, index=False)
            print(f"{file_name} DataFrame successfully exported!")
        else:
            #Si está en blanco generamos un DF con 0 TPV y con la fecha de ejecución
            print(f"El archivo {file_name} está en blanco.")
            data = {"txn_date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "tpv": [0]}
            df_blanco = pd.DataFrame(data)
            df_blanco.to_csv(f"{path_to_save}_2hourly_hist_" + file_name, index=False)
            print(f"{file_name} blank DataFrame successfully exported!")
    
def create_features(df):
    """
    Create time series features from a datetime index.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the datetime index.

    Returns:
    pandas.DataFrame: A new DataFrame with added time series features.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Create a 'date' column and populate it with the datetime index
    df['date'] = df.index

    # Create an 'hour' column and populate it with the hour extracted from the 'date' column
    df['hour'] = df['date'].dt.hour

    # Create a 'dayofweek' column and populate it with the day of the week (0 = Monday, 6 = Sunday)
    df['dayofweek'] = df['date'].dt.dayofweek

    # Create a 'quarter' column and populate it with the quarter of the year
    df['quarter'] = df['date'].dt.quarter

    # Create a 'month' column and populate it with the month (1 = January, 12 = December)
    df['month'] = df['date'].dt.month

    # Create a 'year' column and populate it with the year as a 64-bit integer
    df['year'] = df['date'].dt.year.astype("int64")

    # Create a 'dayofyear' column and populate it with the day of the year
    df['dayofyear'] = df['date'].dt.dayofyear

    # Create a 'dayofmonth' column and populate it with the day of the month
    df['dayofmonth'] = df['date'].dt.day

    # Create a 'weekofyear' column and populate it with the week number of the year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype("int64")

    return df


import pandas as pd

def create_lookback(dataset, look_back=14):
    """
    Crea un DataFrame extendido con datos pasados (look-back) como características.

    Esta función toma un DataFrame de datos de series temporales y crea un nuevo DataFrame
    con características que representan los valores pasados de la serie temporal.

    Parámetros:
    - dataset (DataFrame): El DataFrame de datos de series temporales, con una columna de valores temporales.
    - look_back (int): El número de períodos pasados que se utilizarán como características. Por defecto, es 14.

    Retorna:
    - DataFrame: Un nuevo DataFrame con características generadas a partir de los valores pasados.

    Ejemplo de uso:
    >>> df = pd.DataFrame({'tpv': [10, 15, 20, 25, 30, 35, 40]})
    >>> new_df = create_lookback(df, look_back=3)
    >>> print(new_df)
       tpv  3-day_tpv  2-day_tpv  1-day_tpv
    3   25       10.0       15.0       20.0
    4   30       15.0       20.0       25.0
    5   35       20.0       25.0       30.0
    6   40       25.0       30.0       35.0
    """
    # Genera nombres de columnas para las características basadas en look-back
    lb_names = [str(i) + "-day_tpv" for i in range(1, look_back + 1)]
    lb_names.reverse()  # Invierte el orden para que coincida con los días anteriores

    # Usa shift para crear características de look-back y reemplaza NaN por cero
    for i, col_name in enumerate(lb_names):
        dataset[col_name] = dataset['tpv'].shift(periods=i + 1)
    dataset.fillna(0, inplace=True)
    print("Dataset for lookbacks")
    print(dataset.head(50))
    

    return dataset


#Queries por cliente

client_list = ['OEB1410238K9',
        'AMT170314IP0',
        'PIG930806GL1',
        'OAH091124PR3',
        'VSA180711KM6',
        'SCA070119MQ3',
        'CDJ2011111U4',
        'NME101203MB9',
        'SDI121109B14',
        'PPA0708207I3',
        'CER060901G16',
        'PJU190215RN2',
        'JUM051123KYA',
        'CJF941024F7A',
        'MLS020424LM2',
        'ANL171011549',
        'PGG091023290',
        'OJG130610HG2',
        'IEA141216KQ1',
        'SFA8706028C6']

print(f"Se generarán {len(client_list)} queries, de acuerdo al input.")
#queries_df = generate_queries(client_list)
#queries = queries_df.query_str.to_list()

#print(f"Se generaron un total de {len(queries)} queries para ser procesadas.")


#db = "operacion_bi"

path = r'C:\Users\csanchez_billpocket\Desktop\Billpocket\Data scientist\time_series_model_predictions_tester\data_for_predict\\'
#for element in range(len(queries_df)):
#    print(f"Creando query {queries_df.razon_social[element]}")
#    query_generator(queries_df["query_str"][element], db, f"{path}{queries_df.razon_social[element]}_data_hist.csv")
    

print("Data succesfully obtained!")

#Transformar df por cliente 

# Patrón para buscar archivos CSV (extension .csv)
patron_csv = '*.csv'
# Usar glob para encontrar archivos CSV en la carpeta
archivos_csv = glob.glob(os.path.join(path, patron_csv))
# Imprimir la lista de nombres de archivos CSV encontrados
files = [os.path.basename(archivo) for archivo in archivos_csv]

# Llama a la función con la lista de nombres de archivos
df_creator_hourly(files)
print("Data succesfully transformed!")


#Cargar modelos por cliente

loaded_model = load_model("model_jumamex.h5")
print("Model successfully loaded!")

#Crear predicciones por cliente

data = pd.read_csv("C:\\Users\\csanchez_billpocket\\Desktop\\Billpocket\\Data scientist\\time_series_model_predictions_tester\\two_hourly_data\\_2hourly_hist_CV DIRECTO.csv",
                   parse_dates=["txn_date"], index_col="txn_date")

print("Data original:")
print(data)

data_v1 = data.copy()
print("Data copy:")
print(data_v1)
#creamos los features de fecha y lookback
data_v1 = create_features(data_v1)
print("Data with features:")
print(data_v1)
data_v1 = data_v1.drop(columns=["hour","quarter"])
data_v1 = create_lookback(data_v1)
print("Data with lookbacks:")
print(data_v1)
print("Features correctamente creados.")

# Cargamos la información historica
data_hist = pd.read_csv("bandas_cv_directo.csv", index_col = 'txn_date')
print("Información de bandas historica:")
print(data_hist)


# Crear un objeto MinMaxScaler para estandarizar los datos en el rango [0, 1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Estandarizar las características y etiquetas del conjunto de entrenamiento
data_v1[["14-day_tpv", "7-day_tpv", "1-day_tpv", "tpv"]] = min_max_scaler.fit_transform(data_v1[["14-day_tpv", "7-day_tpv", "1-day_tpv", "tpv"]])

# Crear conjuntos de características y etiquetas para el conjunto de entrenamiento
X, y = data_v1[["14-day_tpv", "7-day_tpv", "1-day_tpv"]], data_v1["tpv"]

print("Elementos X y y generados corresctamente.")
print("X:")
print(X)
print("y:")
print(y)

y_backup = y.copy()

# Formatear los datos de entrada para que coincidan con la forma esperada por el modelo
X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))



#Crear DF con bandas y real
predictions = np.reshape([i[0][0] for i in loaded_model.predict(X)], (-1, 1))

# Las predicciones pueden tener una forma diferente, según el modelo, así que puedes imprimir para verificar
print("Predictions first element:")
print(predictions[0][0])

# Crear un DataFrame que contiene los resultados de las predicciones y las bandas de confianza

# Inicialmente, creamos un DataFrame 'd' con columnas para las predicciones y bandas de confianza
zero_data = np.zeros(shape=(X.shape[0], 4))
d = pd.DataFrame(zero_data, columns=["14-day_tpv", "7-day_tpv", "1-day_tpv", "tpv_predict"])


# Llenar la columna 'tpv_predict' con las predicciones del modelo
d["tpv_predict"] = predictions  # No es necesario reshape aquí
print("Dataframe d antes del inverse_transform en predictions")
print(d)

# Invertir la escala de las predicciones para obtener los valores originales
d = min_max_scaler.inverse_transform(d)
print("Inverse transform")
d = pd.DataFrame(d, columns=["14-day_tpv", "7-day_tpv","1-day_tpv","tpv_predict"])
bandas = d.drop(columns = ["14-day_tpv", "7-day_tpv","1-day_tpv"])
print("DataFrame bandas después del inverse transform en predict")
print(bandas)

#Agregamos el indice de fechas:
bandas = bandas.set_index(y_backup.index)
print("DataFrame bandas después del inverse transform en predict, con indices y con info:")
print(bandas)



print("La información historica luce así:")
print(data_hist)


# Concatenar 'bandas' debajo de 'data_hist' y llenar con ceros las columnas faltantes
bandas = pd.concat([data_hist, bandas], axis=0, sort=False).fillna(0)

# Eliminar duplicados en el índice 'txn_date' (manteniendo el último valor)
bandas = bandas[~bandas.index.duplicated(keep='last')]

# Mostrar el resultado
print("Bandas añadiendo las últimas predicciones:")
print(bandas)



# Calcular las bandas de confianza como la media más/menos una desviación estándar de las predicciones

std_dev = bandas["tpv_predict"].std()  # Calcular la desviación estándar
print(f"Valor de desviación estándar: {std_dev}")
bandas["lim_inf"] = bandas["tpv_predict"] - std_dev
bandas["lim_sup"] = bandas["tpv_predict"] + std_dev

print("DataFrame bandas con limites:")
print(bandas)


# Agregar los valores reales 'tpv_real' del conjunto de prueba
zero_data = np.zeros(shape=(X.shape[0], 4))
d = pd.DataFrame(zero_data, columns=["14-day_tpv", "7-day_tpv", "1-day_tpv", "tpv_real"])
d["tpv_real"] = y.values.reshape(-1, 1)  # No es necesario reshape aquí

# Invertir la escala de las predicciones para obtener los valores originales
d = min_max_scaler.inverse_transform(d)

d = pd.DataFrame(d, columns=["14-day_tpv", "7-day_tpv","1-day_tpv","tpv_real"])

d = d.set_index(y_backup.index)
print("Inverse transform en real con indices y eliminando columnas inecesarias:")
d = d.drop(["14-day_tpv", "7-day_tpv","1-day_tpv"], axis = 1)
print(d)

bandas.update(d)

# Mostrar el resultado
print("bandas con el tpv_real actualizado ")
print(bandas)



# Ahora puedes seguir trabajando con bandas y tpv_real

bandas["25_percentile"] = bandas["tpv_predict"].multiply(0.25)

# Crear una máscara booleana para identificar las filas donde lim_inf <= 0
mask = bandas['lim_inf'] <= 0
print("Elementos por sustituir:")
print(mask.value_counts())

# Sustituir los valores de 'lim_inf' por los valores de '25_percentile' en las filas seleccionadas por la máscara
bandas.loc[mask, 'lim_inf'] = bandas.loc[mask, '25_percentile']


# Determinar si los valores reales están dentro de las bandas de confianza y crear la columna 'In'
bandas["in_flag"] = bandas.apply(lambda row: True if (row.tpv_real >= row.lim_inf and row.tpv_real <= row.lim_sup) else False, axis=1)

print("Elemento dentro y fuera de bandas:")
print(bandas["in_flag"].value_counts())
print((1 - bandas["in_flag"].value_counts()[0]/bandas["in_flag"].count())*100)

print("Elementos negativos en lim_inf:")
print(bandas[bandas["lim_inf"] <= 0])

# Mostrar el DataFrame
print("Resultados finales")
print(bandas)



file_name = 'bandas_historic.csv'
bandas.to_csv(file_name, index = True, encoding='utf-8-sig',)
print(f"Archivo {file_name} correctamente creado!")