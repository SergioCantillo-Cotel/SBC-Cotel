import streamlit as st
from datetime import datetime
from utils import tools
import pandas as pd
import pytz

st.set_page_config(page_title="Smart Building Control: Cotel", layout="wide")
tools.load_custom_css()
st.logo("images/cotel-logotipo.png", size="Large")

pages = [
    st.Page("pages/SBC_Mon.py", title="Cotel - La Flora", icon="🏛️"),
    st.Page("pages/SBC_Cont.py", title="Cotel - La Flora", icon="🏛️")
]

pg = st.navigation({"📈 Monitoreo": [pages[0]],"❄️ Climatización": [pages[1]]}, position="top")
pg.run()

zona = pytz.timezone("America/Bogota")
ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""<div class="footer">🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}</div>""", unsafe_allow_html=True)