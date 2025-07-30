import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import polars as pl
from utils import tools
import gc
import time

RUTA_BMS = 'BMS/Prog_BMS.xlsx'
NOW = pd.Timestamp.now().floor('15min')
START = NOW - pd.Timedelta(weeks=1)

def display_extern_cond(datos, lat=None, lon=None):
    start = time.time()
    with st.container(border=False, key='ext-cond'):
        st.markdown("#### 🌤️ Condiciones Externas")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_custom_metric(col1, "🌡️ Temperatura", f"{datos['T2M'][-1]:.1f} °C")
            st.markdown('<br>', unsafe_allow_html=True)
        with col2:
            render_custom_metric(col2, "💧 Hum. Relativa", f"{datos['RH2M'][-1]:.1f} %")
        with col3:
            render_custom_metric(col3, "🌧️ Precipitaciones", f"{datos['PRECTOTCORR'][-1]:.1f} mm")
    print(f"Condiciones Externas Listas: ejecutada en {time.time() - start:.4f} s")


def display_intern_cond(occup, consumo):
    start = time.time()
    df_cons = consumo.filter(pl.col("unique_id") == "General")
    max_ds = occup.select(pl.col("ds").max()).item()
    df_pers = occup.filter(pl.col("ds") == max_ds)

    with st.container(border=False, key='int-cond'):
        st.markdown("#### 🏭 Condiciones Internas")
        col1, col2 = st.columns(2)
        with col1:
            total_personas = df_pers.select(pl.col("value").sum()).item()
            render_custom_metric(col1, "👥 Personas", f"{total_personas:.0f}")
            st.markdown('<br>', unsafe_allow_html=True)
        with col2:
            consumo = df_cons["value"][-1]
            render_custom_metric(col2, "⚡Consumo", f"{consumo:.1f} kWh")
    print(f"Condiciones Internas Listas: ejecutada en {time.time() - start:.4f} s")


def display_BMS_adj_sch(ruta_excel):
    df = pd.read_excel(ruta_excel, sheet_name="Raw")
    # Crear columna de hora:minuto para mayor resolución
    df["HORA_MIN"] = df["HORA"].astype(str).str.zfill(2) + ":" + df["MIN"].astype(str).str.zfill(2)
    df["DIA_SEMANA"] = df["DIA_SEMANA"].map({
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }).astype(pd.CategoricalDtype(
        categories=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], ordered=True
    ))
    # Pivotear usando hora_min en vez de solo hora
    tabla = df.pivot(index='DIA_SEMANA', columns='HORA_MIN', values='INTENSIDAD')
    custom_text = [[f"Día: {dia}<br>Hora: {hora}<br>Intensidad: {tabla.loc[dia, hora]:.1f}%"
                    for hora in tabla.columns] for dia in tabla.index]

    fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns, y=tabla.index,colorscale='rdbu', 
        colorbar=dict(title='Intensidad',title_font=dict(family='Poppins', color='black'),tickfont=dict(family='Poppins', color='black')),
        text=custom_text, hoverinfo='text',xgap=1,ygap=1))

    fig.update_layout(
        margin=dict(t=10, b=0, l=0, r=0), template='simple_white',height=450, font=dict(family='Poppins', color='black'),
        xaxis=dict(title='Hora', showgrid=True, tickfont=dict(family='Poppins', color='black')),
        yaxis=dict(title='Día', showgrid=True, tickfont=dict(family='Poppins', color='black'))
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    return tabla

def render_custom_metric(col, label, value, delta=None,color='#6c757d',sym=""):
    html = f"""<div class="custom-metric"><div class="label">{label}</div><div class="value">{value}</div>"""
    if delta:
        delta = f"{sym+delta}"
        html += f"""<div class="delta" style="color:{color};">{delta}</div>"""
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)

@st.cache_data(show_spinner="Calculando comparativa...")
def calcular_comparativa_cached(db_AA, db_pers, db_t_ext, db_t_int):
    return tools.calcular_comparativa(db_AA, db_pers, db_t_ext, db_t_int)

def display_comparativa(db_AA, db_pers, db_t_ext=None, db_t_int=None):
    sch_BMS, sch_RT, sch_IA, dif_BMS_RT, dif_BMS_IA = calcular_comparativa_cached(db_AA, db_pers, db_t_ext, db_t_int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sch_BMS["ds"], y=sch_BMS["INTENSIDAD"], mode="lines", name='Prog. BMS', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=sch_RT["ds"], y=sch_RT["value"], mode="lines", name='Prog. Dinámica + Sugerencias IA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sch_IA["ds"], y=sch_IA["intensidad_IA"], mode="lines", name='IA', line=dict(color='green')))    
    fig.update_layout(title="", margin=dict(t=30, b=0, l=20, r=20), font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.99], title="Fecha", showline=True, linecolor='black', showgrid=False, 
                                 zeroline=False, tickfont=dict(color='black'), title_font=dict(color='black')),
                      yaxis=dict(title="Capacidad de Refrigeración (%)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1, yanchor="top"), height=510)
    st.plotly_chart(fig, use_container_width=True)
    return dif_BMS_RT, dif_BMS_IA, sch_IA

def display_temp_zonal(temp, occup):
    start = time.time()
    df_temp = temp.filter(pl.col("unit") == "°C")
    max_ds = temp.select(pl.col("ds").max()).item()

    df_AA = (temp.filter((pl.col("ds") == max_ds) & (pl.col("unique_id").str.contains(r"Valvula_[13579]$"))).sort("unique_id"))
    df_ocup = (occup.filter((pl.col("ds") == max_ds) & (pl.col("unique_id") != "ocupacion_flotante")))

    zonas = [f"T{i}" for i in range(1, 11)]
    espacios = np.array([6, 21, 26, 30, 15], dtype='int8')

    # Agrupar zonas por pares y calcular promedio de temperatura
    Z = []
    for i in range(0, 10, 2):
        subset = df_temp.filter(pl.col("unique_id").is_in(zonas[i:i+2]))
        Z.append(subset.select(pl.col("value").mean()).item())

    with st.container(border=False, key="temp-zon"):
        st.markdown("#### 📍Zonas Monitoreadas")
        zonas_nombres = [
            'Sala de Juntas <br>Cubículos <br>(P2)', ' Gerencia <br> Depto. TI <br> (P2)',
            'G. Humana <br> EE - Preventa <br>(P1)', 'Contabilidad <br> Sala de Juntas <br> (P1)',
            'Nómina <br> Depto. Jurídico <br> (P1)'
        ]
        cols = st.columns(5, vertical_alignment='bottom')
        for i, col in enumerate(cols):
            cont_AA = col.container(border=True)
            estado_raw = df_AA["value"][i]  # 0 u 1
            estado = ["OFF 🔴", "ON 🟢"][int(estado_raw)]
            color_estado = ["#dc3545", "#28a745"][int(estado_raw)]
            ocup = df_ocup["value"][i]

            cont_AA.markdown(
                f"""<div class="custom-metric"> {zonas_nombres[i]}<br><br>
                <div class="value-mon">🌡️ {Z[i]:.1f} °C <br>👥 {ocup:.0f}/{espacios[i]:.0f} <br> <div style="margin-top: 0.5rem; color: {color_estado}; margin-left: 0.1rem;">❄️ {estado}</div></div>
                </div>""",
                unsafe_allow_html=True
            )
    print(f"Display Zonal Listo: ejecutada en {time.time() - start:.4f} s")

def display_mgen(consumo, rango_ev, fecha_int, t_ext, ocup, intensidad, solar, t_int):
    med_Gen = (consumo.filter((pl.col("unique_id") == "General") & (pl.col("ds") >= pd.Timestamp(rango_ev[0])) & (pl.col("ds") <= pd.Timestamp(rango_ev[1]))).select(["ds", "value"]))
    t_ext = t_ext.filter((pl.col("ds") >= pd.Timestamp(fecha_int)) & (pl.col("ds") <= pd.Timestamp(rango_ev[1])))
    ocup = (ocup.group_by("ds").agg(pl.col("value").sum().alias("Ocupacion")).filter((pl.col("ds") >= pd.Timestamp(fecha_int)) & (pl.col("ds") <= pd.Timestamp(rango_ev[1]))))
    intensidad = pl.from_pandas(intensidad[(intensidad['ds'] >= pd.Timestamp(fecha_int)) & (intensidad['ds'] <= pd.Timestamp(rango_ev[1]))].copy())
    solar = solar.filter((pl.col("ds") >= pd.Timestamp(fecha_int)) &(pl.col("ds") <= pd.Timestamp(rango_ev[1])) &(pl.col("unique_id") == "SSFV")).select(["ds", "value"])
    t_int = t_int.with_columns(pl.col("ds").cast(pl.Datetime).dt.truncate("15m")).filter((pl.col("ds") >= pd.Timestamp(fecha_int)) & (pl.col("ds") <= pd.Timestamp(rango_ev[1])))
    for df in [solar, ocup, t_ext, t_int, intensidad]:
         df = df.with_columns(pl.col("ds").dt.truncate("15m"))
    entradas_DT = (solar.join(ocup, on="ds", how="inner").join(t_ext, on="ds", how="inner").join(t_int, on="ds", how="inner").join(intensidad, on="ds", how="inner")).to_pandas()
    DT = tools.digital_twin(entradas_DT)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=med_Gen["ds"].to_list(), y=med_Gen["value"].round(1).to_list(), mode="lines",name='General',line=dict(color='black')))
    fig.add_trace(go.Scatter(x=DT["ds"], y=round(DT["Dig_Twin"],1), mode="lines",name='Digital Twin',line=dict(color='red')))
    fig.add_vline(x=fecha_int, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(title="", margin=dict(t=30, b=0),font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.95], title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False, title_font=dict(color='black'),tickfont=dict(color='black')),
                      yaxis=dict(title="Consumo (kWh)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top", font=dict(color="black")), height=450)
    st.plotly_chart(fig, use_container_width=True)
    del med_Gen, t_ext, ocup, intensidad, solar, t_int, entradas_DT, DT

def display_smart_control_gen(db_ocup, db_clim, t_int, db_AA=None, db_Pow=None):
    max_ds = db_ocup.select(pl.col("ds").max()).item()
    personas = db_ocup.filter(pl.col("ds") == max_ds).select(pl.col("value").sum()).item()
    personas_zona = db_ocup.filter((pl.col("ds").is_not_null()) & (pl.col("ds") == max_ds) & (pl.col("unique_id") != "ocupacion_flotante"))
    t_ext = db_clim["T2M"][-1]
    zonas = ['Sala de Juntas <br>Cubículos <br>(P2)',
             'Gerencia<br> Área TI <br>(P2)',
             'G. Humana <br> EE - Preventa <br>(P1)',
             'Contabilidad <br> Sala de Juntas <br>(P1)',
             'Nómina <br> Depto. Jurídico <br>(P1)']
    
    with st.container(key="styled_tabs_2"):
        tab1, tab2, tab3 = st.tabs(["Programación IA", "Comparativa", "Evaluación de Impacto"])
                
        with tab1.container(key='cont-BMS-IA'):
            start = time.time()
            dia, pronostico = tools.agenda_bms(RUTA_BMS, pd.Timestamp.now(), personas, t_ext, t_int["mean_value"][-1])
            unidades, vel, resultado, _ = tools.seleccionar_unidades(pronostico,personas_zona,pd.Timestamp.now(),dia)
            st.info(resultado)
            with st.container(key='SBC-IA'):
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    cont_zonas = col.container(border=True)
                    cont_zonas.markdown(f"""<div class="custom-metric">{zonas[i]}</div>""", unsafe_allow_html=True)
                    estado = ["🔴 Apagado", "🟢 Encendido"][unidades[i]]
                    estilo = [cont_zonas.error, cont_zonas.success, cont_zonas.warning][unidades[i]]
                    estilo(estado)

                    fig1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=(vel[i] / 7) * 100 if unidades[i] == 1 else 0,
                        number={'suffix': "%"}, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]},'bar': {'color': 'green', 'thickness': 1},}
                    ))
                    fig1.update_layout(margin=dict(t=0, b=0, l=20, r=30), height=120, font=dict(family="Poppins",color="black"))
                    cont_zonas.plotly_chart(fig1, use_container_width=True, key=f'vel{i}',config={'displayModeBar': False})
            print(f"Programación IA Lista: ejecutada en {time.time() - start:.4f} s")
        
        with tab2.container(key='cont-comparativa'):
            estados_AA = db_AA.filter(pl.col("unique_id").str.contains(r"Valvula_[13579]$")).sort("unique_id")
            fecha_reciente = estados_AA.select(pl.col("ds").max()).item()
            valvulas_ultima_fecha = estados_AA.filter(pl.col("ds") == fecha_reciente)
            valores_por_valvula = (valvulas_ultima_fecha.select([pl.col("unique_id"), pl.col("value").cast(pl.Int32)]).sort("unique_id"))
            valores_lista = valores_por_valvula["value"].to_list()
            print(valores_lista, unidades)
            
            with st.container(key='SBC-graph-com'):
                start = time.time()
                col_A,col_B = st.columns([7, 2],vertical_alignment='center')
                with col_A:
                    dif_BMS_RT, dif_BMS_IA, int_IA = display_comparativa(estados_AA, db_ocup, db_clim, db_AA)
                with col_B:
                    color_ia, color_rt = 'red' if dif_BMS_IA >= 0 else 'green', 'red' if dif_BMS_RT >= 0 else 'green'
                    fig_BMS_IA = go.Figure(go.Indicator(mode="number",value=dif_BMS_IA, align = 'center',
                                                        number={'suffix': "%", 'valueformat': '.2f'}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro IA vs<br>Prog. BMS", 'font': {'size': 14, 'color': 'black'}}))
                    fig_BMS_IA.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins", color=color_ia))
                    col_B.plotly_chart(fig_BMS_IA, use_container_width=True, key='dif_BMS_IA',config={'displayModeBar': False})

                    fig_BMS_RT = go.Figure(go.Indicator(mode="number",value=dif_BMS_RT, align = 'center',
                                                        number={'suffix': "%", 'valueformat': '.2f'}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro Prog. Dinámica + Sug. IA vs<br>Prog. BMS", 'font': {'size': 14, 'color': 'black'}}))
                    fig_BMS_RT.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins",color=color_rt))
                    col_B.plotly_chart(fig_BMS_RT, use_container_width=True, key='dif_BMS_RT',config={'displayModeBar': False})
                    print(f"Comparativa IA lista: ejecutada en {time.time() - start:.4f} s")
    
        with tab3.container(key='cont-impacto'):
            start = time.time()
            with st.container(key='SBC-impacto'):
                col_a, col_b = st.columns([1,1], vertical_alignment='center')
                with col_a:
                    rango_est = col_a.date_input("Periodo de evaluación", (pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()), min_value='2025-06-11', max_value=pd.Timestamp.now() ,key='periodo_estudio')
                with col_b:       
                    fecha_int = col_b.date_input("Fecha de Intervención", pd.Timestamp.now() - pd.Timedelta(days=2), min_value=rango_est[0], max_value=rango_est[1] ,key='fecha_intervencion')
                display_mgen(db_Pow,(rango_est[0], rango_est[1] + pd.Timedelta(days=1)), pd.Timestamp(fecha_int),db_clim[['ds','T2M']],db_ocup[['ds','value']],int_IA,db_Pow,t_int)
                print(f"Comparativa vs Gemelo digital lista: ejecutada en {time.time() - start:.4f} s") 