# Anomalies Detection Model

## Overview

The Anomalies Detection Model is a neural network designed to predict the future Total Payment Volume (TPV) behavior of a group of merchants at two-hour intervals. The script leverages these predictions to estimate a minimum and maximum TPV for each time point. If the actual TPV falls outside this range, a visual alarm is triggered.

## Technologies Used

The project is implemented using a variety of technologies, including:

- **Pandas:** Data manipulation and analysis.
- **LSTM TensorFlow Neural Networks:** Building and training the predictive model.
- **SQL:** Managing and querying databases.
- **AWS Athena:** Serverless query service for data analysis.
- **Matplotlib:** Creating visualizations for analysis.

## Final Product

The culmination of this project is a Power BI dashboard that automatically refreshes every two hours. The dashboard provides a comprehensive view of the predicted and actual TPV, highlighting anomalies that require attention.

## Usage

To run the model and generate predictions, follow these steps:

1. **Data Preparation:**
   - Ensure your dataset is formatted correctly.
   - Preprocess the data using the provided preprocessing scripts.

2. **Model Training:**
   - Execute the training script to train the LSTM TensorFlow Neural Network.

3. **Prediction and Alarm:**
   - Run the prediction script to obtain future TPV predictions.
   - The script will raise a visual alarm if the actual TPV is outside the predicted range.

4. **Power BI Dashboard:**
   - Access the Power BI dashboard for a real-time visualization of TPV anomalies.

## Dependencies

To ensure everything will work properly, install all dependencies included in the requirements file included in the repo:

```bash
!pip install -r requirements.txt
```

# Step-by-step technical explanation.

## Data wrangler

First thing first. We need to get historical data to train the model. In order to achieve this I perform AWS Athena queries iteratively. I could do it using MySQL instead, but due to the large number of rows needed AWS Athena endup being the option with the best performance. I use the **_data_get.py_** script to perform the data wrangler.

### Libraries and enviroment variables.

We'll need pandas and numpy as framework to manupulate data as well as datetime to generate dates used in the script. Also, pyathena is needed to create the conection with AWS Athena and their respective keys : 
- **aws_access_key_id**
- **aws_secret_access_key**
- **aws_session_token**
- **s3_staging_dir**
- **region_name**

All these secrets should definetely be defined using dotenv in a .env file. Unfortunately I haven't enough time to do it.

```bash

import pandas as pd
import numpy as np
from datetime import datetime
from pyathena import connect



# Cargar variables de entorno desde el archivo .env


# Obtener las variables de entorno
aws_access_key_id = "Access-key-de-la-empresa"
aws_secret_access_key = "secret-access-key-de-la-empresa"
aws_session_token = "session-tokeen-de-la-empresa"
s3_staging_dir = "s3-address"
region_name = "region"

```

### Data generation.

I used 3 functions to get the data needed for the project: **_athena_query_**, **_queries_string_generator_** and **_data_get_**.

1. The **athena_query function** takeas the AWS secrets as arguments and a query as string. It returns a DataFrame with output of the query.

```bash
#Data_generator
def athena_query(aws_access: str,
                 aws_secret: str,
                 aws_session_token: str,
                 staging_dir: str,
                 region_name: str,
                 query_string: str):
    """
    Execute an SQL query on Amazon Athena using the provided credentials and configuration,
    and save the query result to a CSV file.

    Parameters:
    aws_access (str): The AWS access key ID.
    aws_secret (str): The AWS secret access key.
    aws_session_token (str): The AWS session token.
    staging_dir (str): The S3 bucket path where query results are staged.
    region_name (str): The AWS region name.
    query_string (str): The SQL query to be executed on Athena.
    file_name (str): The name of the CSV file to save the query result.

    Returns:
    None

    Example:
    aws_access_key_id = '...'
    aws_secret_access_key = '...'
    aws_session_token = '...'
    s3_staging_dir = 's3://...'
    region_name = 'us-east-1'
    query = "...your SQL query..."
    file = 'result.csv'
    athena_query(aws_access_key_id, aws_secret_access_key, aws_session_token, s3_staging_dir,
                 region_name, query, file)
    """
    # Establish a connection to Amazon Athena
    conn_athena = connect(
        aws_access_key_id=aws_access,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_session_token,
        s3_staging_dir=staging_dir,
        region_name=region_name
    )

    # Execute the query and fetch results into a Pandas DataFrame
    df_result = pd.read_sql(query_string, conn_athena)

    # Return the DataFrame
    return(df_result)

#Generador de lista de queries a correr dependiendo del rango de fecha de inicio y fecha de fin
# (fin +1)
```

2. The **queries_string_generator** takes a star date and end date as arguments and generate a list of queries with the WHERE clause changing in groups of 1 month, it works using a simple list of dates (dates_list). The output is a list of queries that can be used as input of the athena query function.

```bash
def queries_string_generator (start_date_input, end_date_input):
    
    start_date = start_date_input
    
    end_date = end_date_input
    print(f"Data from {start_date} to {end_date} will be collected")
    current_date = start_date
    dates_list = []

    while current_date < end_date:
        dates_list.append(current_date)
        year = current_date.year if current_date.month != 12 else current_date.year + 1
        month = current_date.month % 12 + 1
        current_date = datetime(year, month, 1)

    base_query = """
    SELECT t."txn_date", t."tpv", m."client"
    FROM "datalake"."tabla" AS t
    LEFT JOIN "datalake"."tabla" AS m
    ON t."id_merchant" = m."id_merchant"
    WHERE t."estatus" = 'aprobada'
    AND (t."txn_type" = 'venta' OR t."txn_type" = 'devolucion')
    AND m.client IN (
        'lista-clientes')
        AND t."txn_date" >= DATE('{}')
        AND t."txn_date" < DATE('{}');
    """

    queries = []

    for i in range(len(dates_list) - 1):
        start_date_str = dates_list[i].strftime('%Y-%m-%d')
        end_date_str = dates_list[i + 1].strftime('%Y-%m-%d')
        query = base_query.format(start_date_str, end_date_str)
        queries.append(query)

    return queries
```

3. The **data_get function** is the combination of both the last two functions it takes a list of queryes for AWS Athena in string format and iterate over all the list excecuting the athena_query function in each iteration. Finally it stores the resulting data frames in .CSV files.  

```bash

#Función que invoca las queries almacenadas en la sección anterior y genera los resultados en archivos .csv generados por mes
def data_get(queries):
    for element in range(len(queries)):
        query_result = athena_query(aws_access_key_id, aws_secret_access_key, aws_session_token, s3_staging_dir,
                region_name, queries[element])
        print(f"Listo elemento {queries[element]} de {len(queries)}")
        query_result.to_csv("data_"+ str(element+1) + ".csv", index = False)
        print("Archivo" + str(element+1) + ".csv" + " correctamente exportado!")
```

Finally the implementation of the three functions.
```bash
start = datetime(2020, 1, 1)
end = datetime(2023, 10, 1)
queries_test = queries_string_generator(start, end)
print(f"{len(queries_test)} queries fueron almacenadas")

data_get(queries_test)

```

## Data transform

As I mentioned, the script is designed to make two-hour interval predictions, the next step is to ransform the raw data into time intervals. Basically I just made a group_by using an interval and client_name. All the data transformation is in the **_data_transform.py_** script.

```bash
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

```
## Data consolidation.
Then, with the **_data_consolid.py_** script I took all the .CSV files generated for the 2-hourly intervals and concat them in a single file called _full_2_hourly.csv_
```bash
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


```