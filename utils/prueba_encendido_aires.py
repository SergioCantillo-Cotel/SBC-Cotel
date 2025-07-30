import math
import numpy as np
import pandas as pd
import holidays, datetime

def agenda_bms(ruta, fecha, num_personas, temp_ext, temp_int):
    df = pd.read_excel(ruta, usecols=[0, 1, 2, 3], names=['dia', 'hora', '_', 'intensidad'])
    dia_str = fecha.strftime('%A')
    dias_es = {'Monday': 'Lunes','Tuesday': 'Martes','Wednesday': 'Miércoles',
               'Thursday': 'Jueves','Friday': 'Viernes','Saturday': 'Sábado',
               'Sunday': 'Domingo'}
    dia_S = dias_es[dia_str]
    h = fecha.hour
    festivos = holidays.CountryHoliday('CO', years=fecha.year)

    if fecha.date() in festivos:
        return dia_S, 0  # <-- Retorna dos valores

    base = df.query("dia == @dia_str and hora == @h")['intensidad']
    if base.empty:
        return dia_S, 0  # <-- Retorna dos valores

    b = base.iat[0]
    ajuste_personas = (-100 if num_personas < 5 else
                       -50 if num_personas < 10 else
                       -25 if num_personas < 20 else
                       0 if num_personas < 40 else
                       25 if num_personas < 50 else 50)

    pron = max(0, min(100, b - (25 - temp_ext) + 1.5 * (temp_int - 25) + ajuste_personas))
    return dia_S, pron


def seleccionar_unidades(pred, personas, fecha, dia):
    zonas = personas[(personas["ds"] == personas["ds"].max()) & (personas["unique_id"] != 'Flotantes')]
    print(zonas)
    if zonas['value'].sum() != 0:
        # Primer cálculo provisional (no se usa para encendido)
        zonas['capacidad'] = np.array([6, 21, 26, 30, 15])
        zonas['proporcion_ocup'] = zonas['value'] / zonas['value'].sum()
        zonas['disponibilidad'] = round(100 - (zonas['value'] / zonas['capacidad'] * 100), 2)
        zonas = zonas.sort_values(by='proporcion_ocup', ascending=False)
        print("zonas:", zonas)
    else:
        zonas['proporcion_ocup'] = 0
        zonas['disponibilidad'] = 100

    aires_ini = math.ceil(pred / 20) if pred > 0 else 0
    print("aires sin corregir:",aires_ini)
    zonas_aire = zonas.head(int(aires_ini)).copy()
    no_encendidos = (zonas_aire['disponibilidad'] == 100).sum()
    carga = round(max(0, pred - (20 * no_encendidos)),2)
    aires = math.ceil(carga / 20) if carga > 0 else 0
    print("aires corregidos:",aires)

    zonas['encendido'] = 0
    if aires > 0:
        zonas.loc[zonas.head(int(aires)).index.intersection(zonas[zonas['disponibilidad'] < 100].index), 'encendido'] = 1

    velocidad_valor = np.ceil(7 * (1 - zonas['disponibilidad'] / 100))
    velocidad_valor = np.clip(velocidad_valor, 0, 7).astype(int)
    
    zonas['velocidad_ventilador'] = (zonas['encendido'] * velocidad_valor).astype(int)
    zonas = zonas.sort_values(by='unique_id', ascending=True)
    #rint(zonas)

    if zonas['encendido'].sum() == 0:
        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que los aires acondicionados en las distintas zonas estén apagados de acuerdo con las condiciones de control."
        )
        encendidas = [0, 0, 0, 0, 0]
    else:
        encendidas = zonas['encendido'].values.tolist()
        zonas_encendidas = zonas[zonas['encendido'] == 1][['unique_id', 'capacidad', 'value']]
        if not zonas_encendidas.empty:
            zonas_lista = (
                "\n" +
                '\n'.join(
                    [
                        f"- {uid}: {100 * (cap - ocup) / cap:.2f}% de capacidad disponible"
                        for uid, ocup, cap in zip(zonas_encendidas['unique_id'], zonas_encendidas['value'], zonas_encendidas['capacidad'])
                    ]
                )
            )
        else:
            zonas_lista = "(No hay zonas encendidas)"
        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que sean encendidos los aires en las siguientes zonas de acuerdo con las condiciones de control:"
            f"{zonas_lista}\n\n"
            "Invitamos a las personas que se encuentren en otros espacios y quieran disfrutar de un mayor confort "
            "a ubicarse en estos lugares y disfruten de la compañía de la familia Cotel."
        )
    return encendidas, zonas['velocidad_ventilador'].values.tolist(), mensaje, carga


ruta = "BMS\programacion_bms.xlsx"
fecha = datetime.datetime.now()
num_personas = 45
temp_int = 25.3
temp_ext = 20.6

dia, resultado = agenda_bms(ruta, fecha, num_personas, temp_ext, temp_int)
aires_on, vel_aires, mensaje, carga = seleccionar_unidades(resultado,
    pd.DataFrame({
        'ds': [fecha.date(), fecha.date(), fecha.date(), fecha.date(), fecha.date()],
        'unique_id': ['Zona 1', 'Zona 2', 'Zona 3', 'Zona 4', 'Zona 5'],
        'value': [0, 6, 14, 17, 8]
    }), fecha, dia
)
print(dia + ":", "\nIntensidad Inicial:", resultado)
print("Intensidad despues de correccion:", carga)
print("Aires Encendidos:", aires_on)
print("Velocidad de Aires:", vel_aires)
print("Recomendación:", mensaje)