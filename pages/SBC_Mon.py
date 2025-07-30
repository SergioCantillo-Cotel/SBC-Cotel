import streamlit as st
from utils import tools, viz
import time

start = time.time()
tools.quarter_autorefresh()
credentials = tools.bigquery_auth()
db_pow, db_temp, db_occup = tools.read_bq_db(credentials)
lat, lon = 3.4793949016367822, -76.52284557701176
datos_cl = tools.get_climate_data(lat, lon)
P1LA = st.sidebar.page_link("http://192.168.5.200:3000/", label="Piso 1 - Lado A", icon="📍")
P1LB = st.sidebar.page_link("http://192.168.5.200:3000/Piso_1_Lado_B", label="Piso 1 - Lado B", icon="📍")
P2 = st.sidebar.page_link("http://192.168.5.200:3000/Piso_2", label="Piso 2", icon="📍")

col1, col2 = st.columns([3, 2], vertical_alignment='bottom')
with col1:
    viz.display_extern_cond(datos_cl, lat, lon)
with col2:
    viz.display_intern_cond(db_occup, db_pow)

viz.display_temp_zonal(db_temp, db_occup)
print(f"Carga de modulo de monitoreo completa: ejecutada en {time.time() - start:.4f} s") 