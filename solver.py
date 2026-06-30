import re
import math
import numpy as np

try:
    from scipy import signal
except Exception:
    signal = None

try:
    import sympy as sp
except Exception:
    sp = None


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def _clean_text(text):
    text = text.replace(",", ".")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("′", "'")
    text = text.replace("´", "'")
    return text


def _numbers(text):
    text = _clean_text(text)
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return [float(v) for v in vals]


def _fmt(x, nd=4):
    try:
        if isinstance(x, complex):
            if abs(x.imag) < 1e-8:
                return f"{x.real:.{nd}f}"
            return f"{x.real:.{nd}f} {'+' if x.imag >= 0 else '-'} {abs(x.imag):.{nd}f}j"
        x = float(x)
        if abs(x) >= 1000:
            return f"{x:,.{nd}f}"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _get_first(patterns, text, default=None):
    text = _clean_text(text)
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return default


def _safe_positive(x, default):
    try:
        x = float(x)
        if x > 0:
            return x
        return default
    except Exception:
        return default


def _basic_result(case, model, formula_latex="", warnings=None):
    return {
        "case": case,
        "model": model,
        "formula_latex": formula_latex,
        "metrics": {},
        "procedure": [],
        "interpretations": [],
        "plot": None,
        "warnings": warnings or []
    }


def _plot_xy(x, y, title, xlabel, ylabel):
    return {
        "x": list(np.asarray(x, dtype=float)),
        "y": list(np.asarray(y, dtype=float)),
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel
    }


# ============================================================
# CLASIFICADOR
# ============================================================

def classify_problem(text, mode):
    m = mode.lower()

    if "newton" in m or "enfriamiento" in m:
        return "newton"

    if "torricelli" in m or "vaciado" in m:
        return "torricelli"

    if "mezcla" in m:
        return "mixtures"

    if "rl" in m or "circuito" in m:
        return "rl"

    if "transferencia" in m:
        return "transfer"

    if "laplace" in m or "primer orden" in m:
        return "first_order"

    t = text.lower()

    if any(w in t for w in ["enfriamiento", "temperatura", "rodamiento", "ambiente", "newton"]):
        return "newton"

    if any(w in t for w in ["torricelli", "vaciar", "vaciado", "drenaje", "orificio"]):
        return "torricelli"

    if any(w in t for w in ["mezcla", "desengrasante", "concentrada", "kg/l", "sale del tanque"]):
        return "mixtures"

    if any(w in t for w in ["inductancia", "corriente", "ohm", "Ω", "voltaje", "circuito", "ri"]):
        return "rl"

    if any(w in t for w in ["función de transferencia", "transferencia", "g(s)", "polos", "ceros"]):
        return "transfer"

    if any(w in t for w in ["'", "derivada", "laplace", "x(t)", "h(t)", "v(t)", "t(t)", "y(t)"]):
        return "first_order"

    return "first_order"


# ============================================================
# CASO 1: LEY DE NEWTON
# ============================================================

def solve_newton(text):
    text = _clean_text(text)
    warnings = []

    temp_vals = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*°?\s*c", text, flags=re.IGNORECASE)
    temp_vals = [float(v) for v in temp_vals]

    min_vals = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(?:min|minuto|minutos)", text, flags=re.IGNORECASE)
    min_vals = [float(v) for v in min_vals]

    if len(temp_vals) >= 4:
        T0 = temp_vals[0]
        Tamb = temp_vals[1]
        T1 = temp_vals[2]
        Ttarget = temp_vals[3]
    else:
        nums = _numbers(text)
        T0 = nums[0] if len(nums) > 0 else 150
        Tamb = nums[1] if len(nums) > 1 else 25
        T1 = nums[3] if len(nums) > 3 else 100
        Ttarget = nums[-1] if len(nums) > 4 else 40
        warnings.append("Algunos datos se asumieron por defecto porque el texto no los dejó claros.")

    t1 = min_vals[0] if min_vals else _get_first([r"tras\s+([-+]?\d+(?:\.\d+)?)"], text, 10)

    if T0 == Tamb or T1 == Tamb or Ttarget <= Tamb:
        return _basic_result(
            "Ley de Newton - Enfriamiento",
            "No se pudo resolver porque las temperaturas no permiten aplicar el modelo.",
            warnings=["Revisa que T0, Tamb, T1 y la temperatura objetivo sean coherentes."]
        )

    k = -math.log((T1 - Tamb) / (T0 - Tamb)) / t1
    t_target = -math.log((Ttarget - Tamb) / (T0 - Tamb)) / k
    t_additional = t_target - t1

    tmax = max(t_target * 1.15, t1 + 10)
    t = np.linspace(0, tmax, 250)
    T = Tamb + (T0 - Tamb) * np.exp(-k * t)

    res = _basic_result(
        "Ley de Newton - Enfriamiento",
        "Modelo térmico de primer orden. La temperatura se acerca a la temperatura ambiente de forma exponencial.",
        r"T(t)=T_{amb}+(T_0-T_{amb})e^{-kt}",
        warnings
    )

    res["metrics"] = {
        "T0": f"{_fmt(T0, 3)} °C",
        "Tamb": f"{_fmt(Tamb, 3)} °C",
        "k": f"{_fmt(k, 6)} 1/min",
        "Tiempo total": f"{_fmt(t_target, 3)} min",
        "Tiempo adicional": f"{_fmt(t_additional, 3)} min"
    }

    res["procedure"] = [
        f"Se aplica la ley de enfriamiento con T0 = {T0} °C, Tamb = {Tamb} °C y T({t1}) = {T1} °C.",
        f"Se calcula k = -ln((T1 - Tamb)/(T0 - Tamb))/t1 = {_fmt(k, 6)} 1/min.",
        f"Se despeja el tiempo para alcanzar {Ttarget} °C.",
        f"El tiempo total es {_fmt(t_target, 3)} min.",
        f"El tiempo adicional después de la medición es {_fmt(t_additional, 3)} min."
    ]

    res["interpretations"] = [
        "La caída de temperatura no es lineal; al inicio enfría rápido y luego se vuelve más lenta porque disminuye la diferencia con el ambiente.",
        "El valor de k representa la rapidez térmica del enfriamiento: mientras mayor sea k, menor será el tiempo de espera.",
        "El equipo solo debe manipularse cuando alcance la temperatura objetivo, porque antes aún existe riesgo térmico para el operario."
    ]

    res["plot"] = _plot_xy(t, T, "Enfriamiento según Ley de Newton", "Tiempo", "Temperatura")
    return res


# ============================================================
# CASO 2: TORRICELLI
# ============================================================

def solve_torricelli(text):
    text = _clean_text(text)
    t_low = text.lower()
    warnings = []

    g = _get_first([r"gravedad\s*(?:como|=)?\s*([-+]?\d+(?:\.\d+)?)", r"g\s*=\s*([-+]?\d+(?:\.\d+)?)"], text, 9.8)
    C = _get_first([r"c\s*=\s*([-+]?\d+(?:\.\d+)?)", r"contracci[oó]n\s*c\s*=\s*([-+]?\d+(?:\.\d+)?)"], text, 0.6)

    Ao = _get_first([
        r"[aá]rea\s*(?:del\s*)?(?:orificio|drenaje)?\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)\s*m2",
        r"a[o0]?\s*=\s*([-+]?\d+(?:\.\d+)?)"
    ], text, None)

    if "cono" in t_low or "cónico" in t_low or "conico" in t_low:
        H = _get_first([
            r"([-+]?\d+(?:\.\d+)?)\s*metros?\s*de\s*altura",
            r"altura\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)"
        ], text, 2)

        R = _get_first([
            r"([-+]?\d+(?:\.\d+)?)\s*metros?\s*de\s*radio",
            r"radio\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)"
        ], text, 0.5)

        res = _basic_result(
            "Torricelli - Tanque cónico",
            "Modelo de vaciado con área variable. En un cono con vértice hacia abajo, el radio depende de la altura.",
            r"\frac{dh}{dt}=-\frac{C A_o\sqrt{2gh}}{\pi\left(\frac{R}{H}\right)^2h^2}",
            warnings
        )

        res["metrics"] = {
            "Altura H": f"{_fmt(H, 3)} m",
            "Radio superior R": f"{_fmt(R, 3)} m",
            "Relación geométrica": f"r(h)=({_fmt(R/H, 4)})h"
        }

        res["procedure"] = [
            f"Por semejanza de triángulos: r/h = R/H = {R}/{H}.",
            f"Entonces r(h) = ({_fmt(R/H, 4)})h.",
            "El área del líquido a una altura h es A(h)=π[r(h)]².",
            "Por Torricelli, el caudal de salida es Q=C Ao √(2gh).",
            "Como A(h) dh/dt = -Q, se obtiene la EDO separable del vaciado."
        ]

        if Ao is not None:
            K = C * Ao * math.sqrt(2 * g) / (math.pi * (R / H) ** 2)
            res["metrics"]["Constante K"] = f"{_fmt(K, 6)}"
            res["procedure"].append(f"Con Ao = {Ao}, la ecuación queda dh/dt = -{_fmt(K, 6)} h^(-3/2).")

            h0 = H
            tiempo = (2 / (5 * K)) * (h0 ** 2.5)
            res["metrics"]["Tiempo de vaciado"] = f"{_fmt(tiempo, 3)} s"

            tt = np.linspace(0, tiempo, 250)
            h = np.maximum((h0 ** 2.5 - 2.5 * K * tt), 0) ** (2 / 5)
            res["plot"] = _plot_xy(tt, h, "Vaciado de tanque cónico", "Tiempo", "Altura h")

        else:
            warnings.append("No se encontró área del orificio. Se plantea la EDO, pero no se calcula tiempo total.")

        res["interpretations"] = [
            "El vaciado cónico no baja linealmente porque el área transversal cambia con la altura.",
            "Cuando la altura disminuye, el área disponible también cambia y eso modifica la velocidad de descenso.",
            "La ecuación queda en variables separables, por eso puede integrarse respecto a h y t."
        ]

        res["warnings"] = warnings
        return res

    r = _get_first([
        r"radio\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)\s*m",
        r"tanque\s*tiene\s*un\s*radio\s*de\s*([-+]?\d+(?:\.\d+)?)"
    ], text, 1)

    h0 = _get_first([
        r"altura\s*inicial\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)\s*m",
        r"altura\s*inicial\s*de\s*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)\s*metros?\s*(?:de\s*)?altura\s*inicial"
    ], text, 4)

    if Ao is None:
        Ao = 0.005
        warnings.append("No se encontró área del orificio. Se asumió Ao = 0.005 m².")

    At = math.pi * r ** 2
    tiempo = (2 * At * math.sqrt(h0)) / (C * Ao * math.sqrt(2 * g))

    tt = np.linspace(0, tiempo, 250)
    h = np.maximum((math.sqrt(h0) - (C * Ao * math.sqrt(2 * g) / (2 * At)) * tt), 0) ** 2

    res = _basic_result(
        "Torricelli - Tanque cilíndrico",
        "Modelo de vaciado por orificio inferior usando la ley de Torricelli.",
        r"\frac{dh}{dt}=-\frac{C A_o}{A_T}\sqrt{2gh}",
        warnings
    )

    res["metrics"] = {
        "Área del tanque": f"{_fmt(At, 4)} m²",
        "Área del orificio": f"{_fmt(Ao, 6)} m²",
        "Tiempo de vaciado": f"{_fmt(tiempo, 3)} s",
        "Tiempo de vaciado": f"{_fmt(tiempo / 60, 3)} min"
    }

    res["procedure"] = [
        f"Se calcula el área del tanque: AT = πr² = π({r})² = {_fmt(At, 4)} m².",
        "Se aplica Torricelli para el caudal de salida: Q = C Ao √(2gh).",
        "Como el volumen del cilindro cambia como V = AT h, entonces AT dh/dt = -Q.",
        "Se integra la ecuación diferencial separable desde h0 hasta 0.",
        f"El tiempo total de vaciado es {_fmt(tiempo, 3)} s, equivalente a {_fmt(tiempo/60, 3)} min."
    ]

    res["interpretations"] = [
        "El nivel baja más rápido al inicio porque la presión hidrostática es mayor cuando la altura es grande.",
        "A medida que el tanque se vacía, la altura disminuye y también disminuye la velocidad de salida.",
        "El tiempo depende directamente del área del tanque e inversamente del área del orificio y del coeficiente C."
    ]

    res["plot"] = _plot_xy(tt, h, "Vaciado de tanque cilíndrico", "Tiempo", "Altura h")
    return res


# ============================================================
# CASO 3: MEZCLAS
# ============================================================

def solve_mixtures(text):
    text = _clean_text(text)
    warnings = []

    V0 = _get_first([
        r"inicialmente\s*([-+]?\d+(?:\.\d+)?)\s*litros",
        r"contiene\s*inicialmente\s*([-+]?\d+(?:\.\d+)?)\s*l",
        r"contiene\s*([-+]?\d+(?:\.\d+)?)\s*litros"
    ], text, 500)

    C_in = _get_first([
        r"([-+]?\d+(?:\.\d+)?)\s*kg\s*/\s*l",
        r"concentraci[oó]n\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)"
    ], text, 0.2)

    rates = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*l\s*/\s*min", text, flags=re.IGNORECASE)
    rates = [float(x) for x in rates]

    if len(rates) >= 2:
        qin = rates[0]
        qout = rates[1]
    elif len(rates) == 1:
        qin = rates[0]
        qout = rates[0]
    else:
        qin = 5
        qout = 5
        warnings.append("No se encontraron caudales. Se asumió qin = qout = 5 L/min.")

    t_eval = _get_first([
        r"despu[eé]s\s*de\s*([-+]?\d+(?:\.\d+)?)\s*min",
        r"([-+]?\d+(?:\.\d+)?)\s*minutos?\s*de\s*haber"
    ], text, 60)

    A0 = 0.0

    if abs(qin - qout) < 1e-9:
        V = V0
        A = C_in * qin * V / qout + (A0 - C_in * qin * V / qout) * math.exp(-(qout / V) * t_eval)

        tt = np.linspace(0, t_eval, 250)
        AA = C_in * qin * V / qout + (A0 - C_in * qin * V / qout) * np.exp(-(qout / V) * tt)

        formula = r"\frac{dA}{dt}=q_{in}C_{in}-\frac{q_{out}}{V}A"

        res = _basic_result(
            "Mezclas - Volumen constante",
            "Modelo de mezcla completamente agitada con entrada y salida al mismo caudal.",
            formula,
            warnings
        )

        res["metrics"] = {
            "Volumen": f"{_fmt(V, 3)} L",
            "Concentración entrada": f"{_fmt(C_in, 4)} kg/L",
            "Caudal": f"{_fmt(qin, 3)} L/min",
            f"A({t_eval} min)": f"{_fmt(A, 4)} kg"
        }

        res["procedure"] = [
            f"El volumen se mantiene constante porque qin = qout = {qin} L/min.",
            f"La entrada de soluto es qin·Cin = {qin}·{C_in} = {_fmt(qin*C_in, 4)} kg/min.",
            f"La salida de soluto es qout·A/V = {qout}·A/{V}.",
            "Se plantea la EDO lineal: dA/dt = qin·Cin - (qout/V)A.",
            f"Evaluando en t = {t_eval} min se obtiene A = {_fmt(A, 4)} kg."
        ]

        res["interpretations"] = [
            "La cantidad de soluto aumenta rápido al inicio porque el tanque comienza con agua pura.",
            "Luego el aumento se reduce porque también sale mezcla con soluto por la descarga.",
            "El sistema tiende a una cantidad máxima de equilibrio igual a la concentración de entrada multiplicada por el volumen."
        ]

        res["plot"] = _plot_xy(tt, AA, "Cantidad de soluto en el tanque", "Tiempo", "Cantidad de soluto")
        return res

    # Caso volumen variable: integración numérica simple
    dt = max(t_eval / 1000, 0.01)
    n = int(t_eval / dt) + 1
    tt = np.linspace(0, t_eval, n)
    A = np.zeros_like(tt)
    A[0] = A0

    for i in range(1, n):
        V = V0 + (qin - qout) * tt[i - 1]
        V = max(V, 1e-6)
        dA = qin * C_in - qout * A[i - 1] / V
        A[i] = A[i - 1] + dA * dt

    res = _basic_result(
        "Mezclas - Volumen variable",
        "Modelo de mezcla agitada con entrada y salida a caudales diferentes.",
        r"\frac{dA}{dt}=q_{in}C_{in}-q_{out}\frac{A}{V_0+(q_{in}-q_{out})t}",
        warnings
    )

    res["metrics"] = {
        "Volumen inicial": f"{_fmt(V0, 3)} L",
        "qin": f"{_fmt(qin, 3)} L/min",
        "qout": f"{_fmt(qout, 3)} L/min",
        f"A({t_eval} min)": f"{_fmt(A[-1], 4)} kg"
    }

    res["procedure"] = [
        "Como qin y qout no son iguales, el volumen cambia con el tiempo.",
        "Se usa V(t)=V0+(qin-qout)t.",
        "Se plantea la EDO de mezcla con volumen variable.",
        "Se resuelve numéricamente para obtener la cantidad de soluto en el tiempo indicado."
    ]

    res["interpretations"] = [
        "El volumen variable cambia la concentración interna y por eso la salida de soluto no es constante.",
        "Si entra más de lo que sale, el tanque acumula volumen y soluto.",
        "Si sale más de lo que entra, el modelo solo es válido hasta antes de que el tanque quede vacío."
    ]

    res["plot"] = _plot_xy(tt, A, "Mezcla con volumen variable", "Tiempo", "Cantidad de soluto")
    return res


# ============================================================
# CASO 4: EDO DE PRIMER ORDEN / LAPLACE / ACTUADOR
# ============================================================

def parse_first_order_equation(text):
    txt = _clean_text(text)
    compact = txt.replace(" ", "")

    variable = "y"
    for v in ["x", "h", "v", "T", "y", "i"]:
        if f"{v}'(t)" in compact or f"{v}'" in compact:
            variable = v
            break

    # Forma: x'(t)+2x(t)=0
    pattern = rf"{variable}'(?:\(t\))?\+?([-+]?\d+(?:\.\d+)?)?{variable}(?:\(t\))?=([-+]?\d+(?:\.\d+)?)"
    m = re.search(pattern, compact, flags=re.IGNORECASE)

    if m:
        a = float(m.group(1)) if m.group(1) not in [None, ""] else 1.0
        b = float(m.group(2))
    else:
        # Búsqueda más general para casos tipo h′(t)+0.2h(t)=0.6
        m2 = re.search(r"'\(t\)\+([-+]?\d+(?:\.\d+)?)[a-zA-Z]\(t\)=([-+]?\d+(?:\.\d+)?)", compact)
        if m2:
            a = float(m2.group(1))
            b = float(m2.group(2))
        else:
            a = 1.0
            b = 0.0

    y0 = _get_first([
        rf"{variable}\(0\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"inicial(?:mente)?\s*(?:en\s*)?(?:reposo|vac[ií]o).*?\(?0\)?",
    ], txt, None)

    if y0 is None:
        if "reposo" in txt.lower() or "vacío" in txt.lower() or "vacio" in txt.lower():
            y0 = 0.0
        else:
            nums = _numbers(txt)
            y0 = 0.0
            for idx, n in enumerate(nums):
                if abs(n) < 1e-12:
                    y0 = n
                    break

    return variable, a, b, y0


def solve_first_order(text):
    text = _clean_text(text)
    low = text.lower()

    if "actuador" in low and ("tau" in low or "τ" in low or "constante de tiempo" in low):
        return solve_actuator_tau(text)

    var, a, b, y0 = parse_first_order_equation(text)

    if abs(a) < 1e-12:
        return _basic_result(
            "EDO de primer orden",
            "No se pudo resolver porque el coeficiente de la variable dependiente es cero.",
            warnings=["Revisa la ecuación. Debe tener forma y'(t)+a y(t)=b."]
        )

    yss = b / a

    tmax = _get_first([
        r"despu[eé]s\s*de\s*([-+]?\d+(?:\.\d+)?)",
        r"t\s*=\s*([-+]?\d+(?:\.\d+)?)"
    ], text, None)

    if tmax is None:
        tmax = max(10 / abs(a), 5)

    tt = np.linspace(0, tmax, 250)
    yy = yss + (y0 - yss) * np.exp(-a * tt)

    final_value = yss + (y0 - yss) * math.exp(-a * tmax)

    res = _basic_result(
        "Laplace / EDO lineal de primer orden",
        "Modelo lineal de primer orden. Se puede resolver por Laplace o por factor integrante.",
        rf"{var}'(t)+{a}{var}(t)={b}",
        []
    )

    res["metrics"] = {
        "Variable": var,
        "a": _fmt(a, 4),
        "Entrada b": _fmt(b, 4),
        "Valor inicial": _fmt(y0, 4),
        "Estado estable": _fmt(yss, 4),
        f"{var}({tmax})": _fmt(final_value, 4)
    }

    res["procedure"] = [
        f"Se identifica la ecuación: {var}'(t) + {a}{var}(t) = {b}.",
        f"Aplicando Laplace: s{var.upper()}(s) - {var}(0) + {a}{var.upper()}(s) = {b}/s.",
        f"Se despeja {var.upper()}(s).",
        f"Al regresar al dominio del tiempo: {var}(t) = {yss} + ({y0} - {yss})e^(-{a}t).",
        f"El valor evaluado en t = {tmax} es {_fmt(final_value, 4)}."
    ]

    if a > 0:
        estabilidad = "estable"
        motivo = "porque el término exponencial e^(-at) desaparece cuando el tiempo aumenta."
    else:
        estabilidad = "inestable"
        motivo = "porque el término exponencial crece con el tiempo."

    res["interpretations"] = [
        f"El sistema es {estabilidad} {motivo}",
        "El valor de estado estable indica a qué valor tiende la variable después de un tiempo prolongado.",
        "La rapidez de respuesta depende de a: mientras mayor sea a, más rápido se estabiliza."
    ]

    res["plot"] = _plot_xy(tt, yy, f"Respuesta de {var}(t)", "Tiempo", var)
    return res


def solve_actuator_tau(text):
    text = _clean_text(text)

    Yss = _get_first([
        r"=\s*([-+]?\d+(?:\.\d+)?)",
        r"valor\s*final\s*(?:de|=)?\s*([-+]?\d+(?:\.\d+)?)"
    ], text, 10)

    yobs = _get_first([
        r"alcanza\s*(?:los\s*)?([-+]?\d+(?:\.\d+)?)",
        r"observa\s*que.*?([-+]?\d+(?:\.\d+)?)\s*cm"
    ], text, 6.32)

    tobs = _get_first([
        r"exactamente\s*([-+]?\d+(?:\.\d+)?)\s*seg",
        r"([-+]?\d+(?:\.\d+)?)\s*segundos"
    ], text, 8)

    tau_opt = _get_first([
        r"[τt]au\s*=\s*([-+]?\d+(?:\.\d+)?)\s*s",
        r"óptimo\s*debe\s*tener\s*una\s*[τt]?\s*=\s*([-+]?\d+(?:\.\d+)?)"
    ], text, 4)

    if yobs >= Yss:
        return _basic_result(
            "Actuador hidráulico",
            "No se pudo calcular tau porque la posición observada debe ser menor al valor final.",
            warnings=["Revisa yobs y el valor final del actuador."]
        )

    tau = -tobs / math.log(1 - yobs / Yss)

    tt = np.linspace(0, max(5 * tau, tobs * 1.5), 250)
    yy = Yss * (1 - np.exp(-tt / tau))

    res = _basic_result(
        "Actuador hidráulico - Constante de tiempo",
        "Modelo de primer orden para desplazamiento de actuador hidráulico.",
        r"\tau\frac{dy}{dt}+y=Y_{ss}",
        []
    )

    res["metrics"] = {
        "Valor final": f"{_fmt(Yss, 3)} cm",
        "Posición observada": f"{_fmt(yobs, 3)} cm",
        "Tiempo observado": f"{_fmt(tobs, 3)} s",
        "Tau calculado": f"{_fmt(tau, 4)} s",
        "Tau óptimo": f"{_fmt(tau_opt, 4)} s"
    }

    res["procedure"] = [
        f"El actuador parte de y(0)=0 y tiende a Yss = {Yss} cm.",
        "La respuesta de primer orden es y(t)=Yss(1-e^(-t/tau)).",
        f"Se reemplaza y({tobs}) = {yobs} cm.",
        f"Se despeja tau = -t/ln(1-y/Yss) = {_fmt(tau, 4)} s.",
        f"La función queda: y(t) = {Yss}(1-e^(-t/{_fmt(tau, 4)}))."
    ]

    if tau > tau_opt:
        diagnosis = "El actuador responde más lento que el valor óptimo."
        cause = "Puede existir fricción interna, fuga hidráulica, aceite muy viscoso, aire en el sistema o desgaste de sellos."
    else:
        diagnosis = "El actuador responde dentro o por debajo del tiempo óptimo."
        cause = "No se observa retraso grave según la constante de tiempo calculada."

    res["interpretations"] = [
        diagnosis,
        cause,
        "Una tau mayor significa respuesta lenta porque el sistema tarda más en acercarse a su posición final."
    ]

    res["plot"] = _plot_xy(tt, yy, "Respuesta del actuador hidráulico", "Tiempo", "Posición")
    return res


# ============================================================
# CASO 5: CIRCUITO RL
# ============================================================

def solve_rl(text):
    text = _clean_text(text)

    L = _get_first([
        r"L\s*=\s*([-+]?\d+(?:\.\d+)?)\s*H",
        r"inductancia.*?([-+]?\d+(?:\.\d+)?)"
    ], text, 1)

    R = _get_first([
        r"R\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:ohm|Ω|O)",
        r"resistencia.*?([-+]?\d+(?:\.\d+)?)"
    ], text, 5)

    V = _get_first([
        r"V\s*=\s*([-+]?\d+(?:\.\d+)?)\s*V",
        r"voltaje.*?([-+]?\d+(?:\.\d+)?)"
    ], text, 110)

    i0 = _get_first([
        r"i\(0\)\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"inicialmente.*?corriente.*?([-+]?\d+(?:\.\d+)?)"
    ], text, 0)

    tau = L / R
    iss = V / R

    tmax = max(5 * tau, 1)
    tt = np.linspace(0, tmax, 250)
    ii = iss + (i0 - iss) * np.exp(-(R / L) * tt)

    res = _basic_result(
        "Circuito RL - Respuesta de corriente",
        "Modelo eléctrico de primer orden para una inductancia y resistencia en serie.",
        r"L\frac{di}{dt}+Ri=V",
        []
    )

    res["metrics"] = {
        "L": f"{_fmt(L, 4)} H",
        "R": f"{_fmt(R, 4)} Ω",
        "V": f"{_fmt(V, 4)} V",
        "Tau": f"{_fmt(tau, 5)} s",
        "Corriente final": f"{_fmt(iss, 4)} A"
    }

    res["procedure"] = [
        f"Se plantea L di/dt + Ri = V con L={L}, R={R}, V={V}.",
        f"El estado estable es i∞ = V/R = {_fmt(iss, 4)} A.",
        f"La constante de tiempo es tau = L/R = {_fmt(tau, 5)} s.",
        f"La solución es i(t) = {iss} + ({i0} - {iss})e^(-{R/L}t).",
        "La corriente sube de forma exponencial hasta aproximarse a V/R."
    ]

    res["interpretations"] = [
        "La inductancia se opone al cambio brusco de corriente, por eso la corriente no alcanza su valor final instantáneamente.",
        "La constante tau indica qué tan rápido desaparece el transitorio eléctrico.",
        "Un valor alto de L o bajo de R hace más lenta la respuesta del circuito."
    ]

    res["plot"] = _plot_xy(tt, ii, "Corriente en circuito RL", "Tiempo", "Corriente")
    return res


# ============================================================
# CASO 6: FUNCIÓN DE TRANSFERENCIA
# ============================================================

def _poly_coeffs(expr):
    expr = _clean_text(str(expr))
    expr = expr.replace("^", "**")

    if sp is None:
        # Permite lista tipo: 1, 3, 2
        vals = _numbers(expr)
        if vals:
            return vals
        return [1.0]

    s = sp.symbols("s")

    try:
        parsed = sp.sympify(expr, locals={"s": s})
        poly = sp.Poly(parsed, s)
        return [float(c) for c in poly.all_coeffs()]
    except Exception:
        vals = _numbers(expr)
        if vals:
            return vals
        return [1.0]


def _roots_to_text(roots):
    if len(roots) == 0:
        return "No tiene"
    return ", ".join([_fmt(r, 4) for r in roots])


def solve_transfer(text, numerator="1", denominator="s^2+3*s+2"):
    warnings = []

    num = _poly_coeffs(numerator)
    den = _poly_coeffs(denominator)

    if len(den) == 0 or all(abs(x) < 1e-12 for x in den):
        return _basic_result(
            "Función de transferencia",
            "Denominador inválido.",
            warnings=["El denominador no puede ser cero."]
        )

    try:
        poles = np.roots(den)
    except Exception:
        poles = np.array([])

    try:
        zeros = np.roots(num) if len(num) > 1 else np.array([])
    except Exception:
        zeros = np.array([])

    stable = all(np.real(p) < 0 for p in poles) if len(poles) > 0 else True

    res = _basic_result(
        "Función de Transferencia",
        "Análisis de polos, ceros, estabilidad y respuesta al escalón.",
        r"G(s)=\frac{Y(s)}{U(s)}=\frac{N(s)}{D(s)}",
        warnings
    )

    res["metrics"] = {
        "Numerador": str(num),
        "Denominador": str(den),
        "Ceros": _roots_to_text(zeros),
        "Polos": _roots_to_text(poles),
        "Estabilidad": "Estable" if stable else "Inestable"
    }

    res["procedure"] = [
        "Se interpreta la función de transferencia como G(s)=Y(s)/U(s).",
        "Los ceros se obtienen igualando el numerador a cero.",
        "Los polos se obtienen igualando el denominador a cero.",
        "La estabilidad se revisa mirando la parte real de los polos.",
        "Si todos los polos tienen parte real negativa, el sistema es estable."
    ]

    if stable:
        res["interpretations"] = [
            "El sistema es estable porque sus polos están en el semiplano izquierdo del plano s.",
            "La respuesta transitoria desaparece con el tiempo y la salida tiende a un comportamiento finito.",
            "Para mantenimiento, esto indica que el sistema absorbe perturbaciones sin crecer indefinidamente."
        ]
    else:
        res["interpretations"] = [
            "El sistema es inestable porque existe al menos un polo con parte real positiva o no negativa.",
            "La salida puede crecer, oscilar sin control o no estabilizarse.",
            "Para mantenimiento, esto indica riesgo operativo y necesidad de revisar control, amortiguamiento o parámetros del sistema."
        ]

    if signal is not None:
        try:
            sys = signal.TransferFunction(num, den)
            t, y = signal.step(sys)
            res["plot"] = _plot_xy(t, y, "Respuesta al escalón", "Tiempo", "Salida")
        except Exception:
            warnings.append("No se pudo graficar la respuesta al escalón con scipy.signal.")
    else:
        warnings.append("scipy no está instalado. Instala scipy para graficar la respuesta al escalón.")

    res["warnings"] = warnings
    return res


# ============================================================
# FUNCIÓN PRINCIPAL USADA POR app.py
# ============================================================

def solve_problem(problem_text, mode="Auto", numerator="1", denominator="s^2+3*s+2"):
    problem_text = problem_text or ""

    kind = classify_problem(problem_text, mode)

    try:
        if kind == "newton":
            return solve_newton(problem_text)

        if kind == "torricelli":
            return solve_torricelli(problem_text)

        if kind == "mixtures":
            return solve_mixtures(problem_text)

        if kind == "rl":
            return solve_rl(problem_text)

        if kind == "transfer":
            return solve_transfer(problem_text, numerator, denominator)

        if kind == "first_order":
            return solve_first_order(problem_text)

        return _basic_result(
            "Caso no reconocido",
            "No se pudo reconocer automáticamente el tipo de problema.",
            warnings=["Selecciona el modo manual desde la barra lateral."]
        )

    except Exception as e:
        return _basic_result(
            "Error de resolución",
            "Ocurrió un error al procesar el problema.",
            warnings=[f"Detalle técnico: {str(e)}"]
        )
