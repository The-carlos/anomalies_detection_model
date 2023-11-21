import pandas as pd
import numpy as np
import glob
import os

# Ruta de la carpeta donde se encuentran los archivos CSV
carpeta = r'directorio\time_series_data_getter\2_hourly_data'

# Patrón para buscar archivos CSV (extension .csv)
patron_csv = '*.csv'

# Usar glob para encontrar archivos CSV en la carpeta
archivos_csv = glob.glob(os.path.join(carpeta, patron_csv))

# Imprimir la lista de nombres de archivos CSV encontrados
files = [os.path.basename(archivo) for archivo in archivos_csv]

df_list = []

for file in files:
    ruta_completa = os.path.join(carpeta, file)
    df_from_file = pd.read_csv(ruta_completa)
    df_list.append(df_from_file)

merged_df = pd.concat(df_list, ignore_index=True)

print(merged_df)

merged_df.to_csv(os.path.join(carpeta, "full_2_hourly.csv"), index = False)
print("Full info successfully exported!")
