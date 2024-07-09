# ============================================================
# Data Extraction
# ============================================================


# Database Path
path_db = "../../data/database/imoveis_prod.db"

# ============================================================
# Libraries Import
# ============================================================

import pandas as pd
import numpy as np
import sqlite3 as sql

# ============================================================
# Connect to the Database
# ============================================================

connect = sql.connect(path_db)

# Making query to export the data to csv
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

# Reading the query into a DataFrame Object
df = pd.read_sql_query(query, connect)


# ============================================================
# Export database to csv
# ============================================================

df.to_csv("../../data/raw/original_dataset.csv", index=None)
