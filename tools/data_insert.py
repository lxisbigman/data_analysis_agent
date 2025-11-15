###data_insert.py
import psycopg2
import pandas as pd

conn_sp = psycopg2.connect(
    host="10.220.77.220",
    port="5432",
    dbname="iguard_db_app",
    user="Digital_2025_3",
    password="Lcdcscy_digital$"
)
query = 'SELECT * FROM network_management.csc_base_info'

conn_ori = psycopg2.connect(
    host="10.5.213.0",
    port="5432",
    dbname="db_ori",
    user="Cloud_2025",
    password="Zqlbmxsf_digital%"
)
query = 'SELECT * FROM public."ori_network_portrait_v1.1"'


portrait = pd.read_sql_query(query, conn_ori)
portrait.to_excel('df_ori.xlsx')

from sqlalchemy import create_engine


engine = create_engine('postgresql://Cloud_2025:Zqlbmxsf_digital%@10.5.213.0:5432/db_sp')

# 确保列名合法、无空格等
portrait.columns = [col.strip() for col in portrait.columns]

portrait.to_sql(
    name='portrait',
    con=engine,
    if_exists='append',     # 表存在就追加，不存在就创建
    index=False,
    method='multi',
    schema='agent'
)
