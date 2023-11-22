
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

#Función que invoca las queries almacenadas en la sección anterior y genera los resultados en archivos .csv generados por mes
def data_get(queries):
    for element in range(len(queries)):
        query_result = athena_query(aws_access_key_id, aws_secret_access_key, aws_session_token, s3_staging_dir,
                region_name, queries[element])
        print(f"Listo elemento {queries[element]} de {len(queries)}")
        query_result.to_csv("data_"+ str(element+1) + ".csv", index = False)
        print("Archivo" + str(element+1) + ".csv" + " correctamente exportado!")

start = datetime(2020, 1, 1)
end = datetime(2023, 10, 1)
queries_test = queries_string_generator(start, end)
print(f"{len(queries_test)} queries fueron almacenadas")

data_get(queries_test)
