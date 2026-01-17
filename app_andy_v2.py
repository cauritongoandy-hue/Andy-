import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fusión de Sensores", layout="wide")

# =========================
# Utilidades
# =========================
def leer_datos(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()

    name = getattr(file, "name", "")
    data = None

    if name.lower().endswith(".csv"):
        data = pd.read_csv(file)
    elif name.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(file)
    else:
        try:
            data = pd.read_csv(file)
        except Exception:
            file.seek(0)
            data = pd.read_excel(file)

    # índice temporal (heurística)
    if data is not None and len(data.columns) > 0:
        time_like = [c for c in data.columns if "time" in c.lower() or "fecha" in c.lower() or "hora" in c.lower()]
        if time_like:
            tcol = time_like[0]
            try:
                data[tcol] = pd.to_datetime(data[tcol], errors="coerce")
                if data[tcol].notna().any():
                    data = data.set_index(tcol)
            except Exception:
                pass
    return data


def grupos_por_correlacion_positive_link(df_T: pd.DataFrame, corr_min: float):
    """
    Agrupa columnas (sensores) exigiendo correlación **positiva** y >= corr_min
    entre cualquier par de sensores de clusters a fusionar (complete-link estricto).
    """
    cols = list(df_T.columns)
    if len(cols) <= 1:
        return [cols] if cols else []

    # Correlación firmada; NaN->0 para impedir uniones inciertas
    C = df_T.corr().fillna(0.0)

    clusters = [{c} for c in cols]

    def min_corr_positive(A, B):
        """Devuelve la mínima correlación entre pares a∈A, b∈B,
        pero si existe algún par con r<=0, falla devolviendo -1.0
        (de modo que no se pueda fusionar)."""
        vals = [C.loc[a, b] for a in A for b in B if a != b]
        if not vals:
            return 1.0
        if any(v <= 0 for v in vals):
            return -1.0
        return min(vals)

    merged = True
    while merged:
        merged = False
        mejor = None
        best_score = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                A, B = clusters[i], clusters[j]
                mc = min_corr_positive(A, B)
                if mc >= corr_min and mc > best_score:
                    best_score = mc
                    mejor = (i, j)
        if mejor is not None:
            i, j = mejor
            clusters[i] = clusters[i].union(clusters[j])
            del clusters[j]
            merged = True

    return [sorted(list(s)) for s in clusters]


def fusionar_grupo_simple(df_T: pd.DataFrame, cols: list) -> pd.Series:
    if not cols:
        return pd.Series(dtype=float, index=df_T.index)
    if len(cols) == 1:
        return df_T[cols[0]].rename(f"{cols[0]}_rep")
    rep = df_T[cols].median(axis=1).rename(f"Tgrp_rep({','.join(cols)})")
    return rep


# =========================
# UI principal
# =========================
st.title("Algoritmo de Fusón de datos")

with st.sidebar:
    st.header("Configuración")
    corr_min = st.slider("Correlación mínima", 0.0, 1.0, 0.85, 0.01)
    uploaded = st.file_uploader("Sube tu dataset", type=["csv", "xlsx", "xls"])

df = leer_datos(uploaded)

if df.empty:
    st.info("Sube un archivo para continuar.")
    st.stop()

df_num = df.select_dtypes(include=[np.number]).copy()
if df_num.shape[1] < 1:
    st.error("No se encontraron columnas numéricas en el archivo.")
    st.stop()

# Pre-procesamiento suave
df_T = df_num.copy().interpolate(limit_direction="both")

# ========== MATRIZ DE CORRELACIÓN (SEABORN) ==========
st.subheader("Matriz de Correlación")
corr = df_T.corr()  # firmado [-1,1]

n = corr.shape[0]
fig_w = min(max(1.0 * n, 10), 28)
fig_h = min(max(0.7 * n, 6), 16)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
sns.heatmap(
    corr,
    ax=ax,
    vmin=-1, vmax=1, center=0,
    cmap="coolwarm",
    annot=True, fmt=".2f",
    annot_kws={"size": 8},
    linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Matriz de Correlación", fontsize=16, pad=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
st.pyplot(fig, clear_figure=True, use_container_width=True)

# =========================
# Agrupar por correlación (SOLO POSITIVA)
# =========================
grupos = grupos_por_correlacion_positive_link(df_T, corr_min=corr_min)
st.subheader("Grupos de fusión (correlación positiva)")
if not grupos:
    st.warning("No se formaron grupos con el umbral dado. Prueba reduciendo 'correlación mínima'.")
    st.stop()

with st.expander("Ver grupos detectados"):
    for k, g in enumerate(grupos, 1):
        st.write(f"Grupo {k}: {', '.join(g)}")

idx = st.number_input("Selecciona el índice de grupo", min_value=1, max_value=len(grupos), value=1, step=1)
cols_grupo = grupos[idx - 1]

st.markdown(f"**Sensores en el grupo seleccionado:** {', '.join(cols_grupo)}")

# =========================
# Paleta de colores fija y contrastada
# =========================
palette = (
    px.colors.qualitative.D3
    + px.colors.qualitative.Bold
    + px.colors.qualitative.Set1
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24
)
color_map = {s: palette[i % len(palette)] for i, s in enumerate(sorted(df_T.columns))}

# =========================
# Gráfica NO FUSIONADOS (apilada)
# =========================
st.markdown("### Sensores del grupo (no fusionados)")
df_plot = df_T[cols_grupo].copy()
df_plot = df_plot.reset_index().rename(columns={df_plot.index.name or 'index': 'Tiempo'})
long_df = df_plot.melt(id_vars="Tiempo", var_name="Sensor", value_name="Valor")
fig_nf = px.line(
    long_df, x="Tiempo", y="Valor", color="Sensor",
    color_discrete_map=color_map,
    labels={"Tiempo": "Tiempo", "Valor": "Valor", "Sensor": "Sensor"}
)
fig_nf.update_layout(legend_title_text="Sensor", hovermode="x unified")
st.plotly_chart(fig_nf, use_container_width=True)

# =========================
# Gráfica FUSIONADO (apilada)
# =========================
st.markdown("### Representante (fusión por mediana)")
rep = fusionar_grupo_simple(df_T, cols_grupo).rename("representante")
df_rep = rep.reset_index().rename(columns={rep.index.name or 'index': 'Tiempo'})
fig_rep = px.line(df_rep, x="Tiempo", y="representante", labels={"Tiempo": "Tiempo", "representante": "Valor"})
fig_rep.update_traces(line=dict(color="black", width=3))
fig_rep.update_layout(showlegend=False)
st.plotly_chart(fig_rep, use_container_width=True)

# =========================
# Resumen
# =========================
st.markdown("### Resumen")
st.json({
    "Número de sensores (totales)": df_T.shape[1],
    "Número de grupos": len(grupos),
    "Tamaño del grupo seleccionado": len(cols_grupo)
})

# =====================================================================
# ===================== BLOQUE KPI (estilo embebida) ===================
# =====================================================================
import numpy as _np
import pandas as _pd

def _hampel_outlier_mask(x: _pd.Series, window: int = 21, n_sigmas: float = 3.5) -> _pd.Series:
    if len(x) == 0:
        return _pd.Series([], dtype=bool, index=x.index)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    xv = x.values.astype(float)
    n = len(xv)
    k = (window - 1) // 2
    med = _np.zeros(n)
    mad = _np.zeros(n)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        win = xv[lo:hi]
        med[i] = _np.nanmedian(win)
        mad[i] = _np.nanmedian(_np.abs(win - med[i])) + 1e-12
    thr = n_sigmas * 1.4826 * mad
    return _pd.Series(_np.abs(xv - med) > thr, index=x.index)

def _rolling_median_mad(x: _pd.Series, win: int = 51):
    half = max(1, win // 2)

    med = x.rolling(win, center=True, min_periods=half).median()

    def _mad(v):
        m = _np.nanmedian(v)
        return _np.nanmedian(_np.abs(v - m))

    mad = x.rolling(win, center=True, min_periods=half).apply(_mad, raw=False)

    # FIX: evitar NaN por MAD=0 y por bordes de ventana
    eps = 1e-12
    mad = mad.fillna(method="bfill").fillna(method="ffill")
    mad = mad.clip(lower=eps)

    return med, mad


def _tukey_weight(z: _pd.Series, c: float = 4.5) -> _pd.Series:
    a = (1 - (z / c) ** 2) ** 2
    w = a.where(_np.abs(z) < c, 0.0)
    return w.clip(0.0, 1.0)

def _soft_range_weight(x: _pd.Series, lo: float, hi: float, span: float = 1.0) -> _pd.Series:
    d = _np.maximum(0.0, _np.maximum(lo - x, x - hi))
    return 1.0 / (1.0 + _np.exp(d - span))

def _stuck_weight(x: _pd.Series, win: int = 41, std_min: float = 0.01) -> _pd.Series:
    half = max(1, win // 2)
    roll_std = x.rolling(win, center=True, min_periods=half).std()
    return 1.0 / (1.0 + _np.exp((std_min - roll_std) / (0.25 * std_min)))

def _coverage_from_contrib(df_g: _pd.DataFrame, incluidos, q_min: float, pot: float):
    Q = df_g[[f"Q__{s}" for s in incluidos]].copy().fillna(0.0)
    M = (Q >= q_min).astype(float)
    W = (Q.clip(lower=0.0) ** pot) * M
    mask = (W.sum(axis=1).values > 0)
    return 100.0 * mask.mean(), mask

df_g = df_T[cols_grupo].copy()
vals_group = df_g.values.reshape(-1)
vals_group = vals_group[_np.isfinite(vals_group)]
if len(vals_group) > 0:
    _lo = float(_np.nanpercentile(vals_group, 1))
    _hi = float(_np.nanpercentile(vals_group, 99))
else:
    _lo = float(df_g.min().min())
    _hi = float(df_g.max().max())

WIN = 101
med_g = _pd.DataFrame(index=df_g.index)
mad_g = _pd.DataFrame(index=df_g.index)
for s in cols_grupo:
    med_g[s], mad_g[s] = _rolling_median_mad(df_g[s], win=WIN)

Z = (df_g[cols_grupo] - med_g[cols_grupo]) / (1.4826 * mad_g[cols_grupo] + 1e-12)
W_band = Z.apply(lambda col: _tukey_weight(col, c=4.5))

out_mask_bin = _pd.DataFrame({
    s: _hampel_outlier_mask(df_g[s], window=21, n_sigmas=3.5).astype(float)
    for s in cols_grupo
})
W_hamp = 1.0 - out_mask_bin

W_rng = _pd.DataFrame({s: _soft_range_weight(df_g[s], _lo, _hi, span=1.0) for s in cols_grupo})
W_stuck = _pd.DataFrame({s: _stuck_weight(df_g[s], win=41, std_min=0.01) for s in cols_grupo})

Q_soft = (W_band * W_hamp * W_rng * W_stuck).clip(0.0, 1.0)
for s in cols_grupo:
    df_g[f"Q__{s}"] = Q_soft[s].astype(float)

_q_min, _pot = 0.5, 1.5
cov, _contrib_mask = _coverage_from_contrib(df_g, cols_grupo, q_min=_q_min, pot=_pot)
loss = 100.0 - cov

st.markdown("""### Indicadores de cobertura (estilo *embebida*)
- **Cobertura utilizable:** {:.1f} %
- **Pérdida de información:** {:.1f} %
""".format(cov, loss))

st.caption("Cobertura = fracción de muestras donde al menos un sensor del grupo aportó realmente (Q≥q_min). No se cuenta el forward-fill.")
st.markdown("**Resumen rápido:** Fusión aplicada solo a sensores con relación **positiva** (r>0 y r≥umbral) dentro del grupo seleccionado.")
