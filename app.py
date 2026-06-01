import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Detección de Fraude", layout="wide")

C = {
    "fondo":   "#edf2f4",
    "oscuro":  "#2b2d42",
    "medio":   "#8d99ae",
    "rojo":    "#780000",
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {C['fondo']}; }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }}
    .kpi-card {{
        background-color: {C['fondo']};
        border-radius: 10px;
        padding: 16px 10px;
        text-align: center;
        border-top: 4px solid {C['oscuro']};
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }}
    .kpi-card.alerta {{ border-top: 4px solid {C['rojo']}; }}
    .kpi-label {{
        font-size: 10px;
        font-weight: 700;
        color: {C['oscuro']};
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 24px;
        font-weight: 700;
        color: {C['oscuro']};
    }}
    .kpi-value.alerta {{ color: {C['rojo']}; }}
    .modelo-card {{
        border-radius: 10px;
        padding: 20px;
    }}
    .modelo-titulo {{
        font-size: 13px;
        font-weight: 700;
        color: {C['oscuro']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    .modelo-subtitulo {{
        font-size: 11px;
        color: {C['medio']};
        margin-bottom: 12px;
    }}
    .modelo-fila {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(43,45,66,0.1);
    }}
    .modelo-label {{
        font-size: 12px;
        color: {C['oscuro']};
    }}
    .modelo-valor {{
        font-size: 15px;
        font-weight: 700;
        color: {C['oscuro']};
    }}
    .modelo-valor.negativo {{ color: {C['rojo']}; }}
    h1 {{
        color: {C['oscuro']} !important;
        font-size: 38px !important;
        text-align: center !important;
        margin-bottom: 4px !important;
        margin-top: 1.5rem !important;
    }}
    p.subtitulo {{
        color: {C['medio']};
        font-size: 13px;
        text-align: center;
        margin-bottom: 1.5rem !important;
    }}
    h3 {{
        color: {C['oscuro']} !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 6px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Datos y modelo ───────────────────────────────────────────
@st.cache_data
def cargar_datos():
    import kagglehub
    import os
    
    # Descargar dataset desde Kaggle
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    archivo = os.path.join(path, 'creditcard.csv')
    df = pd.read_csv(archivo)
    
    # Limpieza
    df = df.drop_duplicates()
    
    # Estandarización
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df['Time'] = df['Time'].astype(int)
    
    return df

@st.cache_data
def entrenar_modelo(df):
    cols_excluir = ['Class', 'Amount', 'Time']
    cols_modelo = [c for c in df.columns if c not in cols_excluir]
    X = df[cols_modelo].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    modelo = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    modelo.fit(X_train, y_train)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    return y_test, y_prob

df = cargar_datos()
y_test, y_prob = entrenar_modelo(df)

y_pred = (y_prob >= 0.7).astype(int)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
tasa_deteccion = tp / (tp + fn)

monto_riesgo = df[df['Class'] == 1]['Amount'].sum()
monto_interceptado = monto_riesgo * tasa_deteccion
monto_no_recuperado = monto_riesgo * (1 - tasa_deteccion)
reduccion = tasa_deteccion * 100

total = len(df)
fraudes = int(df['Class'].sum())
tasa = fraudes / total * 100

# ── Header ───────────────────────────────────────────────────
st.markdown("<h1>Detección de Fraude en Transacciones con Tarjeta de Crédito</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Análisis de 283,726 transacciones reales de tarjetas de crédito europeas (septiembre 2013)</p>", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-label'>Total transacciones</div>
        <div class='kpi-value'>{total:,}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='kpi-card alerta'>
        <div class='kpi-label'>Transacciones fraudulentas</div>
        <div class='kpi-value alerta'>{fraudes:,}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='kpi-card alerta'>
        <div class='kpi-label'>Tasa de fraude</div>
        <div class='kpi-value alerta'>{tasa:.3f}%</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='kpi-card alerta'>
        <div class='kpi-label'>Monto total en riesgo</div>
        <div class='kpi-value alerta'>€{monto_riesgo:,.0f}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Fila 2 ───────────────────────────────────────────────────
col_graf, col_modelo = st.columns([4, 6])

# Gráfico periodos
with col_graf:
    st.markdown("<h3>Fraudes por periodo del día</h3>", unsafe_allow_html=True)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Hora'] = (df['Time'] / 3600) % 24
    df['Periodo'] = pd.cut(df['Hora'],
        bins=[0, 6, 12, 18, 24],
        labels=['Madrugada\n(0h–6h)', 'Mañana\n(6h–12h)',
                'Tarde\n(12h–18h)', 'Noche\n(18h–24h)'],
        include_lowest=True)
    fraudes_periodo = df[df['Class'] == 1].groupby(
        'Periodo', observed=True).size().reset_index(name='Fraudes')

    max_val = fraudes_periodo['Fraudes'].max()

    fig1 = go.Figure(go.Scatter(
        x=fraudes_periodo['Periodo'],
        y=fraudes_periodo['Fraudes'],
        mode='lines+markers+text',
        fill='tozeroy',
        fillcolor="rgba(141,153,174,0.3)",
        line=dict(color=C['oscuro'], width=2.5),
        marker=dict(size=10, color=C['oscuro']),
        text=fraudes_periodo['Fraudes'],
        textposition='top center',
        textfont=dict(color=C['oscuro'], size=13, family='Arial Black')
    ))
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C['oscuro']),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color=C['oscuro']),
            range=[-0.5, 3.5]
        ),
        yaxis=dict(showgrid=False, showticklabels=False,
                   range=[0, max_val * 1.4]),
        margin=dict(t=30, b=20, l=10, r=10),
        height=360
    )
    st.plotly_chart(fig1, use_container_width=True)

# Efectividad del modelo
with col_modelo:
    st.markdown("<div class='modelo-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='modelo-titulo'>Efectividad del modelo</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='modelo-subtitulo'>Evaluado sobre el 20% del dataset · 95 fraudes reales identificados · Umbral de decisión: 0.7</div>", unsafe_allow_html=True)

    col_donut, col_tabla = st.columns([1, 1])

    with col_donut:
        fig2 = go.Figure(go.Pie(
            labels=['Interceptados', 'No detectados'],
            values=[tp, fn],
            hole=0.55,
            marker=dict(colors=[C['oscuro'], C['rojo']]),
            textinfo='label+value',
            textfont=dict(size=11, color=C['fondo']),
            hovertemplate='%{label}: %{value}<extra></extra>'
        ))
        fig2.add_annotation(
            text=f"<b>{reduccion:.0f}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color=C['oscuro'])
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_tabla:
        st.markdown(f"""
        <div style='padding-top: 40px;'>
            <div style='font-size: 12px; color: {C['medio']}; font-style: italic; margin-bottom: 14px;'>
                Estimación basada en la tasa de detección del modelo aplicada al total de fraudes
            </div>
            <div class='modelo-fila'>
                <span class='modelo-label' style='font-size: 14px;'>Monto interceptado</span>
                <span class='modelo-valor' style='font-size: 18px;'>€{monto_interceptado:,.0f}</span>
            </div>
            <div class='modelo-fila'>
                <span class='modelo-label' style='font-size: 14px;'>Monto no recuperado</span>
                <span class='modelo-valor negativo' style='font-size: 18px;'>€{monto_no_recuperado:,.0f}</span>
            </div>
            <div class='modelo-fila'>
                <span class='modelo-label' style='font-size: 14px;'>Reducción de pérdida</span>
                <span class='modelo-valor' style='font-size: 18px;'>{reduccion:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown(f"""
<p style='text-align:center; font-size:10px; color:{C['medio']}; margin-top:10px;'>
Regresión Logística · Umbral de decisión: 0.7 · AUPRC: 0.6720 · 
Datos: Credit Card Fraud Detection — Kaggle (Worldline & ULB)
</p>""", unsafe_allow_html=True)