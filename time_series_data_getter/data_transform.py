import pandas as pd
import numpy as np
import glob
import os


# Ruta de la carpeta donde se encuentran los archivos CSV
carpeta = r'directorio\time_series_data_getter\raw_data'

# Patrón para buscar archivos CSV (extension .csv)
patron_csv = '*.csv'

# Usar glob para encontrar archivos CSV en la carpeta
archivos_csv = glob.glob(os.path.join(carpeta, patron_csv))

# Imprimir la lista de nombres de archivos CSV encontrados
files = [os.path.basename(archivo) for archivo in archivos_csv]

print(files)

def df_creator_hourly(files_names):
    for file_name in files_names:
        file_df = pd.read_csv(os.path.join(carpeta, file_name), parse_dates=['txn_date'], index_col='txn_date')
        # Resetea el índice para evitar conflictos con el nombre 'txn_date'
        file_df.reset_index(inplace=True)
        # Crea una nueva columna 'interval' con los intervalos de 2 horas
        file_df['interval'] = file_df['txn_date'].dt.floor('2H')
        # Agrupa por 'interval' y 'client', y suma 'tpv'
        test_file_data_2hourly = file_df.groupby(['interval', 'client'])['tpv'].sum().reset_index()
        # Renombramos 'interval' a 'txn_date'
        test_file_data_2hourly.rename(columns={'interval': 'txn_date'}, inplace=True)
        print(f"{file_name} successfully grouped by 2-hour intervals and client!")
        # Exporta el DataFrame resultante a un archivo CSV
        test_file_data_2hourly.to_csv("2hourly_" + file_name, index=False)
        print(f"{file_name} DataFrame successfully exported!")

# Llama a la función con la lista de nombres de archivos
df_creator_hourly(files)




