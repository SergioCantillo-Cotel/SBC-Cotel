import streamlit as st
from utils import tools, viz
import time

start = time.time()
tools.quarter_autorefresh()
credentials = tools.bigquery_auth()
db_pow, db_temp, db_occup = tools.read_bq_db(credentials)
lat, lon = 3.4793949016367822, -76.52284557701176
datos_cl = tools.get_climate_data(lat, lon)
t_prom = tools.get_temp_prom(db_temp)
viz.display_smart_control_gen(db_occup, datos_cl, t_prom, db_temp, db_pow)
print(f"Carga de modulo de control SBC: ejecutada en {time.time() - start:.4f} s") 