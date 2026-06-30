import streamlit as st
import matplotlib.pyplot as plt

from solver import solve_problem

st.set_page_config(
    page_title="Resolvedor EDO, Laplace y Transferencia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {background: #F6F8FB;}
    .main-title {
        font-size: 34px; font-weight: 800; color: #111827; margin-bottom: 2px;
    }
    .sub-title {
        font-size: 16px; color: #374151; margin-bottom: 18px;
    }
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 14px;
        border: 1px solid #E5E7EB; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .box {
        background: white; padding: 18px; border-radius: 16px;
        border: 1px solid #E5E7EB; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .small-note {color:#6B7280; font-size: 13px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Resolvedor automático: EDO, Laplace y Función de Transferencia</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Pega el problema, selecciona modo automático o fuerza un tema, y se genera procedimiento, resultado, gráfica e interpretación técnica.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Configuración")

    mode = st.selectbox(
        "Modo de resolución",
        [
            "Auto",
            "Newton - Enfriamiento",
            "Torricelli / Vaciado",
            "Mezclas",
            "Laplace / Primer orden",
            "RL / Circuito eléctrico",
            "Transferencia manual"
        ]
    )

    st.divider()

    st.subheader("Función de transferencia manual")
    st.caption("Puedes escribir coeficientes o polinomios. Ejemplo: num = 1 ; den = s^2+3*s+2")

    numerator = st.text_input("Numerador", value="1")
    denominator = st.text_input("Denominador", value="s^2+3*s+2")

    st.divider()

    st.caption(
        "Temas incluidos: Newton, Torricelli cilíndrico/cónico, mezclas, "
        "actuador hidráulico, Laplace de primer orden, circuito RL, "
        "polos/ceros y respuesta al escalón."
    )

example = """Caso ejemplo:
Un tanque de mezcla contiene inicialmente 500 litros de agua pura. Entra una solución concentrada de desengrasante a 0.2 kg/L con rapidez de 5 L/min. La mezcla sale a la misma razón de 5 L/min. Determina la cantidad de desengrasante después de 60 minutos.
"""

problem = st.text_area(
    "Pega aquí el enunciado completo",
    value=example,
    height=190,
    placeholder="Ejemplo: x'(t)+2x(t)=0, x(0)=0.8 ..."
)

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    solve_btn = st.button("Resolver", type="primary", use_container_width=True)

with c2:
    clear_btn = st.button("Limpiar", use_container_width=True)

if clear_btn:
    st.rerun()

if solve_btn or problem.strip():
    result = solve_problem(problem, mode, numerator, denominator)

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    top1, top2 = st.columns([1.1, 1])

    with top1:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.subheader(result["case"])
        st.write(result["model"])

        if result.get("formula_latex"):
            st.latex(result["formula_latex"])

        st.markdown('</div>', unsafe_allow_html=True)

    with top2:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.subheader("Resultados principales")

        metrics = result.get("metrics", {})

        if metrics:
            for k, v in metrics.items():
                st.metric(k, v)
        else:
            st.info("No se generaron métricas numéricas. Revisa los datos del enunciado o usa modo manual.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Procedimiento")

        for i, step in enumerate(result.get("procedure", []), start=1):
            st.write(f"{i}. {step}")

        st.subheader("Interpretación técnica")

        for item in result.get("interpretations", []):
            st.write(f"- {item}")

    with right:
        st.subheader("Gráfica")

        plot = result.get("plot")

        if plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(plot["x"], plot["y"], linewidth=2)
            ax.set_title(plot.get("title", "Gráfica"))
            ax.set_xlabel(plot.get("xlabel", "x"))
            ax.set_ylabel(plot.get("ylabel", "y"))
            ax.grid(True, alpha=0.35)
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Este caso genera planteamiento simbólico o faltan datos para graficar.")

st.markdown("---")

st.markdown(
    '<div class="small-note">Uso recomendado: pega el problema completo. Si no reconoce el caso, fuerza el tema desde la barra lateral o escribe la función de transferencia manualmente.</div>',
    unsafe_allow_html=True
)
