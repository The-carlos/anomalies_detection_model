
import pandas as pd
import numpy as np
from datetime import datetime
from pyathena import connect
import os
import glob


# Cargar variables de entorno desde el archivo .env


# Obtener las variables de entorno
aws_access_key_id = "ASIAUWHEZMROVMQJWCDP"
aws_secret_access_key = "GdUXM6vSsNoQ0oMXOsiFg4dQG+k7kKVyW/shShU8"
aws_session_token = "IQoJb3JpZ2luX2VjEAoaCXVzLXdlc3QtMiJGMEQCICyvjTe9ik3yu2V98t/j05wbHSPcm7kVgUKpS7TpvWRXAiAjYWzjrfVPcBPvCX8KndaT6Py8SDr3vSmBYUqwWZWFUSqWAwjz//////////8BEAQaDDMyMjYwMjM2MTk0OSIMTXXUwz1xdVUwtEQ+KuoCFeohbyxGcSMNFMOpHyWfRdSyr4zejNn2v7OUW0LpFv32BdQHu0dF9DN/6BjTSk5WC9yaPIdp4Q0vaJWt5U3cOJpNpDaJvy/xmTYv0tyU8lN+SArLmJ+w8OOrcA8ArGd8HEuigweO+AobzLov8nPVbpNBG1tWaEK6KsyxDqIpRXgA5AfzFm4ERoQnStuh+vedXNEC9bbpE/YXp/Ys+CLZ4IWkpyY3qVHKNN49i/fn4rbzB2YyX0WDgsNhEORQAUAiWaaIFEW9sgex1QYFiU8EN/aJ62OAAw0Kt0Cmdij/1bEEFiBi0Oysam9/kN1JrmKUPXPhIaeBONmn30weHowZs1IDH5qqmT4DehLT1DuLnVQgwl9+nQmgJR05cGDyP/DRQqD/gwQOzf+JVQJzIswwC4693MxkH0adfAv7/6j8Uh0qucWYDVWS2lTaCor93lM9jNaWhHvwZ64ho38R+8IBX1iCIrum5HjRPEYwlYyyqAY6pwFd14Ao9MJ2Y+/tWCESp0MZcJymg3LAgBpq/vpU7gf03Vt2jn0KGVc2v59iN/Xtg+ySndcALqKpm8GrkDe86/fhDf8C2qxj4g2FQQpvfqVtYBxA9EdiL2xk+2P4DrK2rDaXsaSalP8fbhspgBxXMtSQ3eydM3hHRc6w6+5j5XX+T9C5Xc0BwyfXwOa71ksiMDp8n4ABV2TqEPhrjZrnFyv9Y4NPAmyxvQ=="
s3_staging_dir = "s3://billpocket-athena-results/"
region_name = "us-east-1"


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
    #start_date = datetime(2023, 3, 1)
    start_date = start_date_input
    #end_date = datetime(2023, 9, 1)
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
    FROM "datalake"."bi_transactions" AS t
    LEFT JOIN "datalake"."bi_merchants" AS m
    ON t."id_merchant" = m."id_merchant"
    WHERE t."estatus" = 'aprobada'
    AND (t."txn_type" = 'venta' OR t."txn_type" = 'devolucion')
    AND m.client IN (
        'OEB1410238K9',
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
        'SFA8706028C6')
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
