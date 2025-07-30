import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import requests_cache, holidays, openmeteo_requests, math, time
from retry_requests import retry
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path

credenciales_json = st.secrets["gcp_service_account"]
BIGQUERY_PROJECT_ID = st.secrets["bigquery"]["project_id"]
BIGQUERY_DATASET_ID = st.secrets["bigquery"]["dataset_id"]
TABLA = st.secrets["bigquery"]["table"]
TABLA_COMPLETA = f"{BIGQUERY_DATASET_ID}.{TABLA}"
CACHE_PATH = "cache/df_power_cache.parquet"
RUTA_BMS = 'BMS/Prog_BMS.xlsx'

def quarter_autorefresh(key: str = "q", state_key: str = "first") -> None:
    """Refresca en el próximo cuarto de hora exacto y luego cada 15 min."""
    ms_to_q = lambda: ((15 - datetime.now().minute % 15) * 60 - datetime.now().second) * 1000 - datetime.now().microsecond // 1000
    first = st.session_state.setdefault(state_key, True)
    interval = ms_to_q() if first else 15 * 60 * 1000
    st.session_state[state_key] = False
    st_autorefresh(interval=interval, key=key)

def bigquery_auth():
    return service_account.Credentials.from_service_account_info(credenciales_json)

def _get_mapping():
    return {
        'PM_General_Potencia_Activa_Total': 'General',
        'PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
        'Inversor_Solar_Potencia_Salida': 'SSFV',
        'ocupacion_sede': 'Ocupacion',
        'ocupacion_flotante': 'Flotantes',
        **{f'IDU_0_{i}_Room_Temperature': f'T{i}' for i in range(1, 11)},
        **{f'IDU_0_{i}_Estado_valvula': f'Valvula_{i}' for i in range(1, 11)},
        'ocupacion_UMA_1': 'Z1: Sala de Juntas - Cubiculos',
        'ocupacion_UMA_2': 'Z2: Gerencia - Area TI',
        'ocupacion_UMA_3': 'Z3: G. Humana - EE - Preventa',
        'ocupacion_UMA_4': 'Z4: Contabilidad - Sala de Juntas',
        'ocupacion_UMA_5': 'Z5: G. Humana - Depto. Jurídico'
    }
def _get_cache(CACHE_PATH, _TYPE):
    if Path(CACHE_PATH).exists():
        df_cache = pl.read_parquet(CACHE_PATH)
        last_date = df_cache.select(pl.col("ds").max()).item()
        where_clause = f"WHERE id IN ({','.join(repr(i) for i in _TYPE)}) AND datetime_record > TIMESTAMP('{last_date}')"
    else:
        df_cache = pl.DataFrame()
        where_clause = f"WHERE id IN ({','.join(repr(i) for i in _TYPE)})"
    return df_cache, where_clause

def read_bq_db(credentials):
    _POWER = ['PM_General_Potencia_Activa_Total', 'PM_Aires_Potencia_Activa_Total', 'Inversor_Solar_Potencia_Salida']
    _TEMP = [f'IDU_0_{i}_Room_Temperature' for i in range(1, 11)] + \
            [f'IDU_0_{i}_Estado_valvula' for i in range(1, 11)] + ['ocupacion_sede']
    _OCCUP = [f'ocupacion_UMA_{i}' for i in range(1, 6)] + ['ocupacion_flotante']
    _ALL_IDS = _POWER + _TEMP + _OCCUP
    _MAPPING = _get_mapping()
    
    df_cache, where_clause = _get_cache(CACHE_PATH, _ALL_IDS)
    client = bigquery.Client(project=BIGQUERY_PROJECT_ID, credentials=credentials)
    sql_query = f"SELECT * FROM `{TABLA_COMPLETA}` {where_clause} ORDER BY datetime_record ASC, id ASC"
    data = [dict(row) for row in client.query(sql_query).result()]

    if not data:
        return df_cache 

    df = pl.DataFrame(data).rename({'id': 'unique_id', 'datetime_record': 'ds'})
    df = df.with_columns(pl.col("unique_id").replace(_MAPPING).alias("unique_id"), pl.col("ds").dt.truncate("15m").alias("ds"))

    df_power = df.filter(pl.col("unique_id").is_in([_MAPPING.get(i, i) for i in _POWER]))
    df_power = gen_others_load(df_power)
    df_AC = df.filter(pl.col("unique_id").is_in([_MAPPING.get(i, i) for i in _TEMP]))
    df_occup = df.filter(pl.col("unique_id").is_in([_MAPPING.get(i, i) for i in _OCCUP]))
    return df_power, df_AC, df_occup

def gen_others_load(df):
    pivot = df.pivot(values="value",index=["ds", "company", "headquarters"],on="unique_id",aggregate_function="first")
    pivot = pivot.with_columns((pl.col("General") + pl.col("SSFV") - pl.col("Aires Acondicionados")).alias("Otros"))
    result = pivot.unpivot(index=["ds", "company", "headquarters"],on=["General", "Aires Acondicionados", "SSFV", "Otros"],variable_name="unique_id",value_name="value").sort(["ds", "unique_id"])
    result = result.with_columns((pl.col("value") * 0.25).round(2).alias("value"))
    return result

def get_climate_data(lat, lon):
    session = retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)
    
    r = client.weather_api("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,"longitude": lon,"models": "gfs_seamless", "timezone": "America/Chicago",
        "minutely_15": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "start_date": "2025-05-15","end_date": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d")})[0].Minutely15()

    start, end = datetime.fromtimestamp(r.Time()), datetime.fromtimestamp(r.TimeEnd())
    interval = timedelta(seconds=r.Interval())
    timestamps = [start + i * interval for i in range((end - start) // interval)]
    df_climate = pl.DataFrame({"ds": timestamps,"T2M": r.Variables(0).ValuesAsNumpy(),"RH2M": r.Variables(1).ValuesAsNumpy(),"PRECTOTCORR": r.Variables(2).ValuesAsNumpy()})
    start_filter, now = datetime(2025, 5, 15, 16, 15), datetime.now()
    df_climate = df_climate.with_columns([(pl.col("ds") - pl.duration(hours=5)).alias("ds")])
    df_climate = df_climate.filter((pl.col("ds") >= start_filter) & (pl.col("ds") <= now - timedelta(hours=5)))
    print(df_climate)
    return df_climate

def load_custom_css(file_path: str = "styles/style.css"):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_temp_prom(db_temp):
    prom = (db_temp.filter(pl.col("unit") == "°C").with_columns([pl.col("ds").dt.truncate("15m"),pl.col("value").cast(pl.Float32)])
            .group_by("ds").agg(pl.mean("value").alias("mean_value")).sort("ds"))
    return prom

def digital_twin(entradas_DT):
    entradas_DT_d = entradas_DT.drop(columns='ds')
    booster = xgb.Booster()
    booster.load_model("IA\modelo_xgb.model")
    dtest = xgb.DMatrix(entradas_DT_d)
    DT = booster.predict(dtest)
    fechas = entradas_DT['ds'].values
    DT = pd.DataFrame({'ds': fechas, 'Dig_Twin': DT})
    #del entradas_DT, dtest, booster
    return DT

def get_prog_bms(inicio, now):
    festivos = holidays.CountryHoliday('CO', years=range(inicio.year, now.year + 1))
    festivos_set = set(festivos.keys())
    df_prog = pl.from_pandas(pd.read_excel(RUTA_BMS, sheet_name='Raw'))
    if 'promedio' in df_prog.columns:
        df_prog = df_prog.drop('promedio')
    if 'MIN' in df_prog.columns:
        df_prog = df_prog.with_columns([pl.col("HORA").cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("hora_h"),
                                        pl.col("MIN").cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("hora_m")]).with_columns(
                                            (pl.col("hora_h") + ":" + pl.col("hora_m")).alias("hora_min")).drop(["hora_h", "hora_m"])
    else:
        df_prog = df_prog.with_columns(pl.format("{:02}:00", pl.col("HORA")).alias("hora_min"))

    cal = pl.DataFrame({"ds": pl.datetime_range(start=inicio, end=now, interval="15m", eager=True)}).with_columns([
        pl.col("ds").dt.strftime("%A").alias("DIA_SEMANA"),
        pl.col("ds").dt.strftime("%H:%M").alias("hora_min"),
        pl.col("ds").dt.date().alias("fecha")
    ])
    sch = cal.join(df_prog, on=["DIA_SEMANA", "hora_min"], how="left")
    sch = sch.with_columns([pl.when(pl.col("fecha").is_in(festivos_set)).then(0).otherwise(pl.col("INT_AIRES_ON")).alias("INTENSIDAD")]).drop("fecha")
    return sch.to_pandas()

def agenda_bms(ruta, fecha, num_personas, temp_ext, temp_int):
    dias_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes","Saturday": "Sábado", "Sunday": "Domingo"
    }
    df = pl.from_pandas(pd.read_excel(ruta, sheet_name='Raw'))
    dia_S = dias_es[fecha.strftime('%A')]
    now = fecha.strftime('%H:%M')

    if 'MIN' in df.columns:
        df = df.with_columns([pl.col("HORA").cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("hora_h"),
                              pl.col("MIN").cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("hora_m")]).with_columns((pl.col("hora_h") + ":" + pl.col("hora_m")).alias("hora_min")).drop(["hora_h", "hora_m"])
    else:
        df = df.with_columns(pl.format("{:02}:00", pl.col("HORA")).alias("hora_min"))

    if 'DIA_SEMANA' in df.columns:
        for eng, esp in dias_es.items():
            df = df.with_columns([pl.when(pl.col("DIA_SEMANA") == eng).then(pl.lit(esp)).otherwise(pl.col("DIA_SEMANA")).alias("DIA_SEMANA")])

    if fecha.date() in holidays.CountryHoliday('CO', years=fecha.year):
        return dia_S, 0

    sel = df.filter((pl.col('DIA_SEMANA') == dia_S) & (pl.col('hora_min') == now))
    if sel.is_empty():
        sel = df.filter((pl.col('DIA_SEMANA') == dia_S) & (pl.col('HORA') == fecha.hour))
    b = sel.select('INTENSIDAD').to_series()[0] if sel.select('INTENSIDAD').height > 0 else 0

    if pd.isna(num_personas) or pd.isna(temp_int):
        pron = b
    else:
        ajuste = (-100 if num_personas < 5 else -50 if num_personas < 10 else -25 if num_personas < 20 else 0 if num_personas < 40 else
                  25 if num_personas < 50 else 50)
        pron = max(0, min(100, b - (25 - temp_ext) + 1.5 * (temp_int - 25) + ajuste))

    return dia_S, pron

def nueva_carga(pred, personas):
    fecha_max = personas.select(pl.col("ds").max()).item()
    zonas = (personas.filter((pl.col("ds") == fecha_max) & (pl.col("unique_id") != 'Flotantes')).with_columns(pl.col("unique_id").cast(pl.String)))

    if zonas["value"].sum() != 0:
        capacidades = [6, 21, 26, 30, 15][:zonas.height]
        zonas = zonas.with_columns([pl.Series("capacidad", capacidades, dtype=pl.Int8)])
        zonas = zonas.with_columns([
            (pl.col("value") / pl.col("value").sum()).alias("proporcion_ocup"),
            ((1 - (pl.col("value") / pl.col("capacidad"))) * 100).round(2).alias("disponibilidad")
        ]).sort("proporcion_ocup", descending=True)
    else:
        zonas = zonas.with_columns([pl.lit(0).alias("proporcion_ocup"),pl.lit(100).alias("disponibilidad")])

    aires_ini = math.ceil(pred / 20) if pred > 0 else 0
    zonas_aire = zonas.head(aires_ini)
    no_encendidos = zonas_aire.filter(pl.col("disponibilidad") == 100).height
    carga = round(max(0, pred - (20 * no_encendidos)), 2)
    aires = math.ceil(carga / 20) if carga > 0 else 0

    zonas = zonas.with_columns([pl.lit(0).alias("encendido")])

    if aires > 0:
        candidatas = zonas.filter(pl.col("disponibilidad") < 100).sort("proporcion_ocup", descending=True).head(aires)
        zonas = zonas.with_columns([pl.when(pl.col("unique_id").is_in(candidatas["unique_id"])).then(1).otherwise(0).alias("encendido")])

    zonas = zonas.with_columns([pl.min_horizontal([pl.max_horizontal([(7 * (1 - pl.col("disponibilidad") / 100)).ceil(),pl.lit(0)]),pl.lit(7)]).cast(pl.Int8).alias("vel_raw")])
    zonas = zonas.with_columns([(pl.col("encendido") * pl.col("vel_raw")).cast(pl.Int8).alias("velocidad_ventilador")]).sort("unique_id")
    total_enc = zonas.select(pl.col("encendido").sum()).item()
    resultado = total_enc * 20
    return float(max(0, resultado))

def seleccionar_unidades(pred, personas, fecha, dia):
    start = time.time()
    # Asegurar que trabajamos en modo eager
    if isinstance(personas, pl.LazyFrame):
        personas = personas.collect()

    # Filtrar últimas zonas válidas
    fecha_max = personas.select(pl.col("ds").max()).item()
    zonas = (personas.filter((pl.col("ds") == fecha_max) & (pl.col("unique_id") != 'Flotantes')).with_columns(pl.col("unique_id").cast(pl.String)))

    if zonas["value"].sum() != 0:
        capacidades = [6, 21, 26, 30, 15][:zonas.height]
        zonas = zonas.with_columns([pl.Series("capacidad", capacidades, dtype=pl.Int8)])
        zonas = zonas.with_columns([
            (pl.col("value") / pl.col("value").sum()).alias("proporcion_ocup"),
            ((1 - (pl.col("value") / pl.col("capacidad"))) * 100).round(2).alias("disponibilidad")
        ]).sort("proporcion_ocup", descending=True)
    else:
        zonas = zonas.with_columns([pl.lit(0).alias("proporcion_ocup"),pl.lit(100).alias("disponibilidad")])

    # Estimación de carga
    aires_ini = math.ceil(pred / 20) if pred > 0 else 0
    zonas_aire = zonas.head(aires_ini)
    no_encendidos = zonas_aire.filter(pl.col("disponibilidad") == 100).height
    carga = round(max(0, pred - (20 * no_encendidos)), 2)
    aires = math.ceil(carga / 20) if carga > 0 else 0

    zonas = zonas.with_columns([pl.lit(0).alias("encendido")])

    if aires > 0:
        candidatas = zonas.filter(pl.col("disponibilidad") < 100).sort("proporcion_ocup", descending=True).head(aires)
        zonas = zonas.with_columns([pl.when(pl.col("unique_id").is_in(candidatas["unique_id"])).then(1).otherwise(0).alias("encendido")])

    zonas = zonas.with_columns([pl.min_horizontal([pl.max_horizontal([(7 * (1 - pl.col("disponibilidad") / 100)).ceil(),pl.lit(0)]),pl.lit(7)]).cast(pl.Int8).alias("vel_raw")])
    zonas = zonas.with_columns([(pl.col("encendido") * pl.col("vel_raw")).cast(pl.Int8).alias("velocidad_ventilador")]).sort("unique_id")
    encendidas = zonas["encendido"].to_list()
    velocidades = zonas["velocidad_ventilador"].to_list()

    if sum(encendidas) == 0:
        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que los aires acondicionados en las distintas zonas estén apagados de acuerdo con las condiciones de control."
        )
        encendidas = [0, 0, 0, 0, 0]  # fallback
    else:
        zonas_encendidas = zonas.filter(pl.col("encendido") == 1)
        if zonas_encendidas.height > 0:
            zonas_lista = "\n" + "\n".join([
                f"- {uid}: {100 * (cap - ocup) / cap:.2f}% de capacidad disponible"
                for uid, ocup, cap in zip(
                    zonas_encendidas["unique_id"],
                    zonas_encendidas["value"],
                    zonas_encendidas["capacidad"]
                )
            ])
        else:
            zonas_lista = "(No hay zonas encendidas)"

        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que sean encendidos los aires en las siguientes zonas de acuerdo con las condiciones de control:"
            f"{zonas_lista}\n\n"
            "Invitamos a las personas que se encuentren en otros espacios y quieran disfrutar de un mayor confort "
            "a ubicarse en estos lugares y disfruten de la compañía de la familia Cotel."
        )
    return encendidas, velocidades, mensaje, carga
    
def calcular_comparativa(db_AA, db_pers, db_t_ext, db_t_int):
    now = pd.Timestamp.now().floor('15min')
    inicio = now - pd.Timedelta(weeks=1)
    t15 = lambda df: df.with_columns(pl.col("ds").dt.truncate("15m")).filter(pl.col("ds").is_between(inicio, now, closed="both"))

    sch_BMS = pl.from_pandas(get_prog_bms(inicio, now))
    fechas_base = pl.DataFrame({'ds': sch_BMS['ds']})

    db_AA_pl = t15(db_AA)
    sch_RT = fechas_base.join(db_AA_pl.group_by('ds').agg((pl.col('value').sum() * 100 / pl.col('value').count()).alias('value')),on='ds',how='left')

    db_pers_pl = t15(db_pers) 
    pz=db_pers_pl.drop_nulls(subset=['ds'])
    db_pers_join = fechas_base.join(db_pers_pl.group_by('ds').agg(pl.col('value').sum().alias('value')),on='ds',how='left')

    db_t_ext_pl = t15(db_t_ext)
    db_t_ext_join = fechas_base.join(db_t_ext_pl.group_by('ds').agg(pl.col('T2M').sum().alias('T2M')),on='ds',how='left')

    db_t_int_pl = db_t_int.filter((pl.col('ds') >= inicio) & (pl.col('ds') <= now) & (pl.col('unique_id').str.strip_chars().str.contains(r'^T(10|[1-9])$'))).with_columns(pl.col('ds').dt.truncate('15m'))
    pivot_mean = (db_t_int_pl.pivot(values="value", index="ds", columns="unique_id", aggregate_function="mean").with_columns(promedio_T=pl.mean_horizontal(pl.all().exclude("ds"))).select(["ds", "promedio_T"]))
    db_t_int_join = fechas_base.join(pivot_mean,on="ds",how="left")

    # --- Optimizado: cálculo de sch_IA vectorizado con Polars (sin apply por fila) ---
    # Construir base con personas (value), temperatura externa (T2M), temperatura interna promedio (promedio_T) e INTENSIDAD BMS
    df_ia_pl = (
        fechas_base.join(db_pers_join.select("ds", "value"), on="ds", how="left")
        .join(db_t_ext_join.select("ds", "T2M"), on="ds", how="left").join(db_t_int_join.select("ds", "promedio_T"), on="ds", how="left")
        .join(sch_BMS.select(["ds", "INTENSIDAD"]), on="ds", how="left")
    )
    df_ia_pl = df_ia_pl.with_columns([pl.when(pl.col("value") < 5).then(-100).when(pl.col("value") < 10).then(-50)
         .when(pl.col("value") < 20).then(-25).when(pl.col("value") < 40).then(0)
         .when(pl.col("value") < 50).then(25).otherwise(50).alias("ajuste")])

    df_ia_pl = df_ia_pl.with_columns(
        (pl.col("INTENSIDAD") - (25 - pl.col("T2M")) + 1.5 * (pl.col("promedio_T") - 25)+ pl.col("ajuste")).clip(0, 100).alias("pron")
    )

    CAP_MAP = {
        "Z1: Sala de Juntas - Cubiculos": 6,"Z2: Gerencia - Area TI": 21,
        "Z3: G. Humana - EE - Preventa": 26,"Z4: Contabilidad - Sala de Juntas": 30,
        "Z5: G. Humana - Depto. Jurídico": 15,
    }

    # Filtrar y enriquecer ocupaciones por zona con su capacidad
    pz_zonas = (
        pz.filter(pl.col("unique_id") != "Flotantes").with_columns(pl.col("unique_id").replace(CAP_MAP).alias("capacidad").cast(pl.Int16))
    )

    # Agregados por timestamp: total de personas, zonas con ocupación > 0, y zonas con disponibilidad (ocupación < capacidad)
    pz_agg = (pz_zonas.group_by("ds").agg([
        pl.col("value").sum().alias("personas_total"),
        (pl.col("value") > 0).cast(pl.Int8).sum().alias("nonzero_count"),
        (pl.col("value") < pl.col("capacidad")).cast(pl.Int8).sum().alias("available_count"),
    ]))

    sch_IA_pl = (df_ia_pl.join(pz_agg, on="ds", how="left")
            # si hay nulls en conteos, llénalos a 0 para evitar propagación
            .with_columns([pl.col("nonzero_count").fill_null(0),pl.col("available_count").fill_null(0),
            ])
            # aires_ini = ceil(pron/20)  ->  floor((pron + 19)/20)   (como pron >= 0)
            .with_columns([
                ((pl.col("pron") + 19) / 20).floor().cast(pl.Int16).alias("aires_ini")
            ])
            # no_encendidos = max(0, aires_ini - nonzero_count)
            .with_columns([
                (pl.col("aires_ini") - pl.col("nonzero_count")).clip(0).alias("no_encendidos")
            ])
            # carga = max(0, pron - 20*no_encendidos)
            .with_columns([
                (pl.col("pron") - 20 * pl.col("no_encendidos")).clip(0).alias("carga")
            ])
            # aires = (carga>0) ? ceil(carga/20) : 0  ->  floor((carga + 19)/20) si carga>0
            .with_columns([
                pl.when(pl.col("carga") > 0).then(((pl.col("carga") + 19) / 20).floor().cast(pl.Int16)).otherwise(0).alias("aires")
            ])
            .with_columns([
                pl.min_horizontal(pl.col("aires"), pl.col("available_count")).alias("encendidos")
            ])
            # intensidad_IA = 20 * encendidos
            .with_columns([
                (20 * pl.col("encendidos")).cast(pl.Float64).alias("intensidad_IA")]).select(["ds", "intensidad_IA"]))

    sch_IA = sch_IA_pl.to_pandas()
    sA = sch_BMS["INTENSIDAD"].cast(pl.Float64)
    sB = sch_RT["value"].cast(pl.Float64)
    sC = pl.from_pandas(sch_IA["intensidad_IA"]).cast(pl.Float64)
    
    mask_RT_BMS = sA.is_not_null() & sB.is_not_null() & (sA != 0)
    mask_RT_IA = sA.is_not_null() & sC.is_not_null() & (sA != 0)

    dif_BMS_RT = (((sA.filter(mask_RT_BMS) - sB.filter(mask_RT_BMS)) / sA.filter(mask_RT_BMS)) * 100).mean()
    dif_BMS_IA = (((sA.filter(mask_RT_IA) - sC.filter(mask_RT_IA)) / sA.filter(mask_RT_IA)) * 100).mean()
  
    return sch_BMS,sch_RT,sch_IA,-dif_BMS_RT, -dif_BMS_IA
