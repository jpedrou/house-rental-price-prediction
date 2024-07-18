# =================================================
# Data Manager
# =================================================


# =================================================
# Libraries Import
# =================================================

import math
import sqlite3 as sql
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

# =================================================
# Definings some constant configs
# =================================================

DEFAULT_ARGS = {"owner": "airflow"}
ROOT_PATH = "/home/joao/Documents/IF/api_streamlit/data/database"
PROCESSED_TMP_DF_PATH = "/home/joao/Documents/IF/api_streamlit/data/processed/"
PRODUCTION_PATH = ROOT_PATH + "/imoveis_prod.db"
DATAWHAREHOUSE_PATH = ROOT_PATH + "imoveis_dw.db"
TMP_DF_PATH = "/home/joao/Documents/IF/api_streamlit/data/raw/original_dataset.csv"


# =================================================
# Dag Configuration
# =================================================

dag = DAG(
    dag_id="data_manager",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(2),
    catchup=False,
)


# =================================================
# Functions to transform the data
# =================================================


def extract():
    connection = sql.connect(PRODUCTION_PATH)
    query = """
    SELECT CIDADE.NOME as 'cidade'
    ,ESTADO.NOME as 'estado'
    ,IMOVEIS.AREA as 'area'
    ,IMOVEIS.NUM_QUARTOS
    ,IMOVEIS.NUM_BANHEIROS
    ,IMOVEIS.NUM_ANDARES
    ,IMOVEIS.ACEITA_ANIMAIS
    ,IMOVEIS.MOBILIA
    ,IMOVEIS.VALOR_ALUGUEL
    ,IMOVEIS.VALOR_CONDOMINIO
    ,IMOVEIS.VALOR_IPTU
    ,IMOVEIS.VALOR_SEGURO_INCENDIO
    FROM IMOVEIS INNER JOIN CIDADE
    ON IMOVEIS.CODIGO_CIDADE = CIDADE.CODIGO
    INNER JOIN ESTADO
    ON CIDADE.CODIGO_ESTADO = ESTADO.CODIGO;
    """
    df = pd.read_sql_query(query, connection)
    df.to_csv(TMP_DF_PATH, index=None)

    connection.close()


def preprocess():
    df = pd.read_csv(TMP_DF_PATH)
    tmp = df.copy()

    # Cidade Column
    tmp["cidade"] = tmp["cidade"].map(
        {
            "São Paulo": 0,
            "Rio de Janeiro": 1,
            "Belo Horizonte": 2,
            "Porto Alegre": 3,
            "Campinas": 4,
        }
    )

    # Estado Column
    tmp["estado"] = tmp["estado"].map({"SP": 0, "RJ": 1, "MG": 2, "RS": 3})

    # num_andares Column
    count_distribution = tmp.loc[tmp["num_andares"] != "-", "num_andares"].value_counts(
        normalize=True
    )

    value = tmp.loc[tmp["num_andares"] == "-", "num_andares"].value_counts().values[0]

    tmp.loc[tmp["num_andares"] == "-", "num_andares"] = np.nan

    for key, number in count_distribution.items():
        dis = math.ceil(value * number)
        tmp["num_andares"].fillna(key, limit=dis, inplace=True)
        value -= dis

    tmp["num_andares"].fillna(tmp["num_andares"].mode()[0], inplace=True)

    # aceita_animais Column
    tmp["aceita_animais"] = tmp["aceita_animais"].map({"acept": 1, "not acept": 0})

    # mobilia Column
    tmp["mobilia"] = tmp["mobilia"].map({"furnished": 1, "not furnished": 0})

    tmp.to_csv(PROCESSED_TMP_DF_PATH + "cleaned_df.csv", index=None)


def handle_outliers():
    df_cleaned = pd.read_csv(PROCESSED_TMP_DF_PATH + "cleaned_df.csv")
    tmp = df_cleaned.copy()

    tmp["area"] = np.where(tmp["area"] > 2000, np.nan, tmp["area"])

    tmp["valor_aluguel"] = np.where(tmp["area"].isna(), np.nan, tmp["valor_aluguel"])

    tmp["valor_condominio"] = np.where(
        tmp["area"].isna(), np.nan, tmp["valor_condominio"]
    )

    tmp["num_andares"] = np.where(tmp["area"].isna(), np.nan, tmp["num_andares"])

    tmp["valor_iptu"] = np.where(tmp["area"].isna(), np.nan, tmp["valor_iptu"])

    tmp["valor_aluguel"] = np.where(
        tmp["valor_aluguel"] > 40000,
        np.nan,
        tmp["valor_aluguel"],
    )

    tmp["valor_iptu"] = np.where(tmp["valor_iptu"] > 30000, np.nan, tmp["valor_iptu"])

    tmp["num_andares"] = np.where(tmp["num_andares"] > 32, np.nan, tmp["num_andares"])

    imputer = KNNImputer(n_neighbors=10, weights="distance")

    tmp[["area", "valor_aluguel", "valor_condominio", "valor_iptu", "num_andares"]] = (
        imputer.fit_transform(
            tmp[
                [
                    "area",
                    "valor_aluguel",
                    "valor_condominio",
                    "valor_iptu",
                    "num_andares",
                ]
            ]
        )
    )

    tmp["num_andares"] = tmp["num_andares"].astype(int)

    tmp.to_csv(PROCESSED_TMP_DF_PATH + "no_outliers_df.csv", index=None)


# =================================================
# Dags Operations
# =================================================

extract_task = PythonOperator(task_id="extract", python_callable=extract, dag=dag)
preprocess_task = PythonOperator(
    task_id="preprocess", python_callable=preprocess, dag=dag
)
handle_outliers_task = PythonOperator(
    task_id="handle_outliers", python_callable=handle_outliers, dag=dag
)


# ETL
extract_task >> preprocess_task >> handle_outliers_task
