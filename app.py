# -*- coding: utf-8 -*-
"""
Herramientas TIC para la Intervención en Trabajo Social — UNA Puno
Ejecutar:   streamlit run app.py
Requisitos: pip install -U streamlit google-genai pypdf
Coloca el PDF de la monografía junto a este archivo (HERRAMIENTAS_TIC_TRABAJO_SOCIAL.pdf).

Cómo logra la respuesta «instantánea»
  1. STREAMING: el texto se muestra mientras Gemini lo genera (generate_content_stream).
  2. VOZ DEL NAVEGADOR: cada frase completa se lee con la Web Speech API (speechSynthesis) apenas
     termina de escribirse. No hay una segunda llamada a la API para el audio ni cuota de TTS.

Configuración: TODO lo técnico (clave, modelos, voz, límites) está en la sección CONFIGURACIÓN de
este archivo. La interfaz solo muestra controles para la persona usuaria.

Heurísticas de usabilidad de Nielsen aplicadas (las marcas «H1…H11» del código indican dónde):
  H1  Visibilidad del estado del sistema · texto en streaming, indicador «Consultando…», barra de voz
      con estado en vivo, estado del documento base, aviso del modelo de respaldo y del filtro PII.
  H2  Coincidencia con el mundo real · lenguaje del Trabajo Social y mensajes en español claro, sin
      jerga técnica (ya no aparecen «API Key», «TTS», «motor»…); íconos reconocibles.
  H3  Control y libertad del usuario · «Detener voz» siempre visible, silenciar la voz, «Nueva
      conversación» con confirmación, barra lateral plegable.
  H4  Consistencia y estándares · mismos botones, etiquetas e íconos en todos los mensajes; patrón
      de chat convencional; colores con significado fijo (rojo = detener/no, verde = sí).
  H5  Prevención de errores · filtro PII activo por defecto (también en la voz), límite de longitud
      de la consulta, controles deshabilitados si el servicio no está configurado, confirmación
      antes de borrar.
  H6  Reconocer en lugar de recordar · preguntas rápidas, controles de voz siempre visibles.
  H7  Flexibilidad y eficiencia · escribir, hablar o elegir una pregunta rápida; velocidad de voz
      regulable; «Escuchar» en cada respuesta.
  H8  Diseño estético y minimalista · la configuración técnica salió de la interfaz; solo quedan
      los controles que la persona usuaria necesita.
  H9  Ayudar a reconocer, diagnosticar y recuperarse de errores · mensajes con causa y acción,
      botón «Reintentar», se conserva la respuesta parcial, detalles técnicos plegados.
  H10 Ayuda y documentación · «¿Cómo usar este asistente?» y textos de ayuda en cada control.
  H11 Accesibilidad e inclusión (heurística añadida) · lectura en voz alta, velocidad regulable,
      foco visible, controles propios de ≥ 44 px, regiones aria-live, respeto a «reducir movimiento».
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

try:
    from google import genai
    from google.genai import types

    GENAI_OK = True
except Exception:  # SDK no instalado: la app sigue funcionando sin el chatbot
    GENAI_OK = False

log = logging.getLogger("tic_trabajo_social")

# ═══════════════════════════ CONFIGURACIÓN (BACKEND) ═══════════════════════════
# Lo técnico se cambia SOLO aquí. Nada de esto aparece en la interfaz.

# ── Servicio de IA (Gemini) ──
# Si la dejas vacía, se busca en st.secrets["GEMINI_API_KEY"] y luego en la variable de entorno
# GEMINI_API_KEY. ⚠️ Si pegas la clave aquí, NO subas este archivo a un repositorio público.
GEMINI_API_KEY = ""
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Se prueban en orden si el modelo elegido está saturado (503/429) o ya no existe (404).
MODELOS_RESPALDO = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")
REINTENTOS_POR_MODELO = 2  # intentos antes de pasar al modelo siguiente
PAUSA_REINTENTO_S = 1.5  # segundos de espera entre intentos
TEMPERATURA: float | None = None  # None = valor por defecto del modelo
HISTORIAL_MENSAJES = 6  # mensajes previos que se reenvían al modelo como contexto
MAX_CONSULTA = 1500  # caracteres máximos por pregunta escrita (H5)

# ── Documento base ──
PDF_NOMBRE = "HERRAMIENTAS_TIC_TRABAJO_SOCIAL.pdf"
MAX_CONTEXTO = 120_000  # caracteres del PDF enviados al modelo
PERMITIR_SUBIR_PDF = False  # True = si falta el PDF local, la barra lateral deja subirlo

# ── Privacidad ──
FILTRO_PII_INICIAL = True  # el filtro (DNI y teléfonos) arranca activado

# ── Voz: Web Speech API del navegador (gratuita, sin API Key) ──
VOZ_INICIAL = True  # lectura automática activada al abrir la app
VOZ_IDIOMA = "es-PE"  # idioma por defecto si el equipo no tiene voces en español
VOZ_IDIOMAS = ("es-PE", "es-US", "es-419", "es-MX", "es-ES")  # orden de preferencia
# Fragmentos de nombre de voces masculinas habituales (Windows, macOS, Edge). Se prefiere la primera que exista.
VOZ_PREFERIDAS = ("Alex", "Raul", "Raúl", "Pablo", "Jorge", "Diego", "Juan", "Carlos", "Alvaro", "Álvaro")
# True  = usa siempre una voz instalada en el equipo si existe: arranca al instante y el texto NO sale del equipo (recomendado con datos de casos).
# False = elige por nombre/idioma aunque sea una voz en línea (p. ej. las «Natural» de Edge: suenan mejor, pero envían el texto a Microsoft).
VOZ_PREFERIR_LOCAL = True
VOZ_TONO = 0.85  # 0.1–2.0 · menos de 1 = voz más grave
VOZ_VELOCIDADES = {"Lenta": 0.85, "Normal": 1.0, "Rápida": 1.25}
VOZ_VELOCIDAD_INICIAL = "Normal"
MAX_FRASE_VOZ = 220  # las frases más largas se parten en comas para empezar a hablar antes
# Cómo debe sonar lo que la voz suele leer mal (expresión regular → texto leído).
PRONUNCIACION = {
    r"\bet al\.": "y colaboradores",
    r"\bNASW\b": "N.A.S.W.",
    r"\bTICs?\b": "tics",
}

# ── Identidad visual y textos institucionales ──
AZUL, PURPURA = "#003366", "#6F42C1"
LOGO_URL = ""  # Pega aquí la URL del escudo de la UNA Puno (o guarda escudo_unap.png junto a app.py)
FACULTAD = "Facultad de Trabajo Social - UNA Puno"
CURSO = "TICs para la Intervención en Trabajo Social"
DOCENTE = "Docente: Sucapuca Tito, Elizabeth · VII semestre · 2026-II"
# ═════════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

st.set_page_config(
    page_title="TIC en Trabajo Social · UNA Puno",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {background:#F7F8FC;}
html, body, [class*="css"] {font-size:17px;}
h1, h2, h3, h4 {color:#003366;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:4px solid #6F42C1;}
.hero {background:linear-gradient(120deg,#003366 0%,#6F42C1 100%); padding:1.4rem 1.8rem; border-radius:16px; margin-bottom:1rem;}
.hero h1 {color:#fff; margin:0; font-size:2rem;}
.hero p {color:#E8E3F7; margin:.3rem 0 0 0; font-size:1.05rem;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {background:#fff; border:1px solid #D9DCEA; border-radius:10px 10px 0 0; padding:.6rem 1.1rem; min-height:44px;}
.stTabs [data-baseweb="tab"] p {color:#003366; font-weight:600; font-size:1.05rem; margin:0;}
.stTabs [aria-selected="true"] {background:#003366;}
.stTabs [aria-selected="true"] p {color:#fff;}
.stTabs [data-baseweb="tab-highlight"] {background:#6F42C1;}
.escudo {width:110px; height:110px; border-radius:50%; background:#003366; color:#fff; border:5px solid #6F42C1;
         display:flex; align-items:center; justify-content:center; text-align:center; font-weight:800; margin:0 auto .6rem auto;}
.nivel {background:#fff; border-radius:14px; padding:1rem 1.2rem; border-top:6px solid var(--c); box-shadow:0 2px 8px rgba(0,0,0,.08); height:100%;}
.badge {display:inline-block; background:var(--c); color:#fff; border-radius:50%; width:2rem; height:2rem; line-height:2rem;
        text-align:center; font-weight:700; margin-right:.5rem;}
.col-si, .col-no {color:#fff; padding:.7rem 1rem; border-radius:12px; font-weight:700; font-size:1.15rem; margin-bottom:.6rem;}
.col-si {background:#0F766E;} .col-no {background:#B42318;}
.pie {text-align:center; color:#4A4A4A; font-size:.85rem; margin-top:1rem;}
/* H11 · foco visible para quien navega con teclado */
:focus-visible {outline:3px solid #6F42C1 !important; outline-offset:2px;}
/* H11 · respeta «reducir movimiento» del sistema */
@media (prefers-reduced-motion: reduce) {* {animation:none !important; transition:none !important; scroll-behavior:auto !important;}}
/* El iframe invisible que habla no debe ocupar espacio */
.stElementContainer:has(iframe[title="Voz del navegador"]), .element-container:has(iframe[title="Voz del navegador"]) {
    position:absolute; width:1px; height:1px; overflow:hidden; opacity:0; pointer-events:none; margin:0;}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── Privacidad (PII) ─────────────────────────
RE_TEL = re.compile(r"(?<!\d)(?:\+?51[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")  # 9 dígitos
RE_DNI = re.compile(r"(?<!\d)\d{8}(?!\d)")  # 8 dígitos


def enmascarar_pii(texto: str) -> tuple[str, int, int]:
    """Reemplaza teléfonos y DNI. Devuelve (texto, n_dni, n_telefonos)."""
    if not texto:
        return texto, 0, 0
    texto, n_tel = RE_TEL.subn("[TELÉFONO_OCULTO]", texto)
    texto, n_dni = RE_DNI.subn("[DNI_OCULTO]", texto)
    return texto, n_dni, n_tel


# ───────────────────────── Lectura del PDF ─────────────────────────
@st.cache_data(show_spinner=False)
def extraer_texto_pdf(datos: bytes) -> tuple[str, int]:
    lector = PdfReader(io.BytesIO(datos))
    paginas = []
    for i, pagina in enumerate(lector.pages, 1):
        try:
            t = pagina.extract_text() or ""
        except Exception:
            t = ""
        t = re.sub(r"[ \t]+", " ", t).strip()
        if t:
            paginas.append(f"[Pág. {i}]\n{t}")
    return "\n\n".join(paginas), len(lector.pages)


@st.cache_data(show_spinner=False)
def cargar_pdf_local(ruta: str, firma: float) -> tuple[str, int]:
    """`firma` (fecha de modificación) invalida la caché si cambia el archivo."""
    return extraer_texto_pdf(Path(ruta).read_bytes())


def localizar_pdf() -> Path | None:
    exacto = BASE_DIR / PDF_NOMBRE
    if exacto.is_file():
        return exacto
    candidatos = sorted(p for p in BASE_DIR.glob("*.pdf") if "HERRAMIENTAS" in p.name.upper())
    return candidatos[0] if candidatos else None


def obtener_contexto() -> tuple[str, str, bool]:
    """Devuelve (texto, mensaje_estado, ok)."""
    ruta = localizar_pdf()
    if ruta:
        try:
            texto, paginas = cargar_pdf_local(str(ruta), ruta.stat().st_mtime)
            if texto.strip():
                return texto, f"📄 {ruta.name} · {paginas} págs.", True
            return "", "El documento base no tiene texto que se pueda leer (¿es un escaneo?).", False
        except Exception as e:
            log.warning("No se pudo leer el PDF: %s", e)
            return "", "No se pudo leer el documento base.", False
    if PERMITIR_SUBIR_PDF:
        subido = st.file_uploader("No encontré el documento base. Súbelo aquí:", type="pdf")
        if subido is not None:
            try:
                texto, paginas = extraer_texto_pdf(subido.getvalue())
                if texto.strip():
                    return texto, f"📄 {subido.name} · {paginas} págs.", True
            except Exception as e:
                log.warning("No se pudo leer el PDF subido: %s", e)
                return "", "No se pudo leer el documento subido.", False
    return "", "Falta el documento base: el asistente responderá solo con conocimiento general.", False


# ───────────────────────── Gemini (con streaming) ─────────────────────────
def clave_api() -> str:
    """La clave se toma del archivo (CONFIGURACIÓN), luego de st.secrets y luego del entorno."""
    if GEMINI_API_KEY.strip():
        return GEMINI_API_KEY.strip()
    try:
        valor = st.secrets.get("GEMINI_API_KEY", "")
        if valor:
            return str(valor).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def construir_sistema(contexto: str, pii: bool) -> str:
    doc = contexto[:MAX_CONTEXTO] if contexto else "(No se cargó el documento base: avisa al usuario que respondes solo con conocimiento general.)"
    privacidad = (
        "El filtro ético de privacidad está ACTIVO: nunca repitas DNI ni teléfonos; usa [DNI_OCULTO] y [TELÉFONO_OCULTO]."
        if pii
        else "Recuerda al usuario que no debe compartir datos reales de casos."
    )
    return f"""Eres el «Asistente de Protocolos TIC» de la Escuela Profesional de Trabajo Social de la UNA Puno.
Responde en español, con tono profesional, claro y accesible (máximo ~250 palabras salvo que pidan más).
Reglas:
1. Basa tus respuestas en el DOCUMENTO BASE y cita a los autores tal como aparecen (Reamer, Castells, van Dijk, Gómez Navarro et al., NASW et al. 2017, FITS 2018, Hwang 2024, etc.).
2. Si algo no está en el documento (p. ej. la Ley N.° 29733), indícalo como conocimiento general y pide verificar la norma vigente.
3. Las TIC complementan, no sustituyen, la empatía, la escucha activa ni el juicio ético profesional.
4. Considera siempre confidencialidad, consentimiento informado y brecha digital (zonas rurales de Puno, lenguas quechua y aimara).
5. {privacidad}
6. Tu respuesta se leerá en voz alta frase por frase: escribe frases completas y no demasiado largas, y evita tablas.

=== DOCUMENTO BASE ===
{doc}
=== FIN DEL DOCUMENTO ==="""


def _es(texto: str, claves: tuple[str, ...]) -> bool:
    return any(k in texto for k in claves)


_CLAVES_RETIRADO = ("404", "not_found", "no longer available")
_CLAVES_SATURADO = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted")


def flujo_gemini(api_key: str, contenido, sistema: str):
    """Generador: va entregando el texto de Gemini a medida que se produce (streaming).

    Si el modelo está saturado o retirado ANTES de emitir texto, reintenta y pasa al respaldo.
    Si falla a mitad de la respuesta no se puede cambiar de modelo: se propaga el error.
    """
    cliente = genai.Client(api_key=api_key)
    candidatos = [MODELO] + [m for m in MODELOS_RESPALDO if m != MODELO]
    extra = {"temperature": TEMPERATURA} if TEMPERATURA is not None else {}
    ultimo_error: Exception | None = None
    for modelo in candidatos:
        for intento in range(REINTENTOS_POR_MODELO):
            emitido = False
            try:
                config = types.GenerateContentConfig(system_instruction=sistema, **extra)
                for trozo in cliente.models.generate_content_stream(model=modelo, contents=contenido, config=config):
                    texto = getattr(trozo, "text", None) or ""
                    if not texto:
                        continue
                    if not emitido:
                        emitido = True
                        st.session_state["modelo_usado"] = modelo
                    yield texto
            except Exception as e:
                if emitido:
                    raise  # ya se mostró texto: no se puede reintentar sin duplicarlo
                t = str(e).lower()
                if _es(t, _CLAVES_RETIRADO):
                    ultimo_error = e  # modelo retirado: pasar al siguiente
                    break
                if _es(t, _CLAVES_SATURADO):
                    ultimo_error = e  # saturación temporal: reintentar y luego pasar al siguiente
                    if intento < REINTENTOS_POR_MODELO - 1:
                        time.sleep(PAUSA_REINTENTO_S)
                    continue
                raise
            if not emitido:
                raise RuntimeError("La respuesta llegó vacía (posible bloqueo de seguridad). Reformula la pregunta.")
            return
    raise ultimo_error or RuntimeError("No hay un modelo de Gemini disponible.")


def filtrar_pii_stream(flujo):
    """Aplica el filtro PII sobre el streaming sin dejar pasar números partidos entre trozos.

    Se retiene la cola numérica (dígitos, espacios, guiones) hasta saber si el número terminó.
    """
    pendiente = ""
    for trozo in flujo:
        pendiente += trozo
        cola = re.search(r"[\d+][\d \-]*$", pendiente)
        seguro, pendiente = (pendiente[: cola.start()], pendiente[cola.start():]) if cola else (pendiente, "")
        if seguro:
            yield enmascarar_pii(seguro)[0]
    if pendiente:
        yield enmascarar_pii(pendiente)[0]


_RE_TRANSCRIPCION = re.compile(r"TRANSCRIPCI[ÓO]N:\s*(.*?)\s*RESPUESTA:\s*", re.S | re.I)


def separar_transcripcion(flujo, al_transcribir):
    """Para consultas por voz: retiene «TRANSCRIPCIÓN: … RESPUESTA:» y deja pasar solo la respuesta."""
    buf, listo, recortar = "", False, False
    for trozo in flujo:
        if listo:
            if recortar:  # el espacio tras «RESPUESTA:» puede llegar en el trozo siguiente
                trozo = trozo.lstrip()
                recortar = not trozo
            if trozo:
                yield trozo
            continue
        buf += trozo
        m = _RE_TRANSCRIPCION.search(buf)
        if m:
            al_transcribir(m.group(1).strip())
            listo = True
            resto = buf[m.end():].lstrip()
            recortar = not resto
            if resto:
                yield resto
        elif len(buf) > 800:  # el modelo no respetó el formato: no hacer esperar más
            al_transcribir("(no disponible)")
            listo = True
            yield buf
    if not listo:
        al_transcribir("(no disponible)")
        if buf.strip():
            yield buf


def mensaje_error(e: Exception) -> str:
    """Mensaje en lenguaje claro: qué pasó y qué puede hacer la persona (H2, H9)."""
    t = str(e).lower()
    if _es(t, ("api key", "api_key", "401", "403", "permission", "unauthenticated")):
        return "🔑 El servicio de IA no está bien configurado (la clave no es válida o no tiene permisos). Avisa a quien administra la aplicación."
    if _es(t, ("429", "quota", "resource_exhausted")):
        return "⏳ Se alcanzó el límite de uso del servicio. Espera un momento e inténtalo de nuevo."
    if _es(t, ("503", "unavailable", "high demand", "overloaded")):
        return "🚦 Los modelos de IA están saturados en este momento. Espera unos segundos y vuelve a intentarlo."
    if _es(t, _CLAVES_RETIRADO):
        return "🧭 El modelo de IA configurado ya no está disponible. Quien administra la aplicación debe actualizarlo en la sección CONFIGURACIÓN de app.py."
    if "vacía" in t or "empty" in t:
        return "🙈 La respuesta llegó vacía (puede ser un bloqueo de seguridad). Reformula la pregunta con otras palabras."
    return "⚠️ No se pudo completar la consulta. Inténtalo de nuevo; si el problema sigue, avisa a quien administra la aplicación."


# ───────────────────────── Voz del navegador (Web Speech API) ─────────────────────────
_RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d]")


def texto_para_voz(texto: str) -> str:
    """Quita Markdown, enlaces, emojis y símbolos para que la voz lea solo lenguaje natural."""
    t = re.sub(r"```.*?```", " ", texto, flags=re.S)
    t = re.sub(r"\[([^\]]+)\]\(https?://[^)]*\)", r"\1", t)  # [texto](url) → texto
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\[P[áa]g\.?\s*\d+(?:\s*[-–,]\s*\d+)*\]", "", t)  # marcas [Pág. 12] del documento base
    t = t.replace("[DNI_OCULTO]", "dato oculto").replace("[TELÉFONO_OCULTO]", "dato oculto")
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)  # títulos
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)  # viñetas
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)  # numeración «1. »
    t = re.sub(r"[*_`~|]+", "", t)  # énfasis y tablas
    t = _RE_EMOJI.sub("", t)
    t = re.sub(r"<[^<>\n]{1,25}>", " ", t)  # <br>, <sigh>…: no deben leerse
    t = t.replace("<", " ").replace(">", " ")
    for patron, lectura in PRONUNCIACION.items():
        t = re.sub(patron, lectura, t)
    t = re.sub(r"(?<!\.)\.\.(?!\.)", ".", t)  # «N.A.S.W.» + punto final no debe quedar con doble punto
    lineas = []
    for linea in t.splitlines():
        linea = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", linea)).strip()
        if linea:  # una pausa natural al final de cada línea
            lineas.append(linea if linea[-1] in ".,;:!?…)" else linea + ".")
    return "\n".join(lineas)


# Fin de oración: puntuación (+ comillas/paréntesis/énfasis de cierre) + espacio + inicio de otra oración.
# Exige ver la letra siguiente: mientras llega el streaming, una oración no se da por cerrada «a medias».
_RE_FIN_ORACION = re.compile(r"([.!?…]+[\"»)\]*_]*)[ \t]+(?=[A-ZÁÉÍÓÚÑÜ¿¡«\"(\[*#•\-])")
# Abreviaturas y numeraciones («1. ») tras las cuales NO termina la oración.
_RE_ABREVIATURA = re.compile(
    r"(?:\bet al|\bp\. ?ej|\bpágs?|\bN|\bDra?|\bLic|\bSra?|\bMg|\bIng|\bProf|\bart|\bcap|\bvol|\bed|\bej|(?<![^\s])\d{1,2})\.$",
    re.I,
)


def _cortes(texto: str) -> list[int]:
    return [m.end(1) for m in _RE_FIN_ORACION.finditer(texto) if not _RE_ABREVIATURA.search(texto[: m.end(1)])]


def _piezas(texto: str, cortes: list[int]) -> list[str]:
    piezas, ini = [], 0
    for c in cortes:
        piezas.append(texto[ini:c])
        ini = c
    piezas.append(texto[ini:])
    return piezas


def extraer_frases(buf: str, final: bool = False) -> tuple[list[str], str]:
    """Separa el texto acumulado en (frases completas, resto sin terminar). Con final=True no deja resto."""
    *lineas, resto = buf.split("\n")
    crudas: list[str] = []
    for linea in lineas:  # un salto de línea siempre cierra la frase (párrafos y viñetas)
        crudas += _piezas(linea, _cortes(linea))
    if final:
        crudas += _piezas(resto, _cortes(resto))
        resto = ""
    else:
        cortes = _cortes(resto)
        if cortes:
            crudas += _piezas(resto, cortes)[:-1]
            resto = resto[cortes[-1]:].lstrip()
    return [c.strip() for c in crudas if c.strip()], resto


def trocear(frase: str, limite: int = MAX_FRASE_VOZ) -> list[str]:
    """Parte una frase muy larga en comas/punto y coma para que la voz empiece antes y no se corte."""
    salida: list[str] = []
    while len(frase) > limite:
        corte = max(frase.rfind(s, 0, limite) for s in (", ", "; ", ": "))
        if corte < limite // 3:
            corte = frase.rfind(" ", 0, limite)
        if corte <= 0:
            corte = limite - 1
        salida.append(frase[: corte + 1].strip())
        frase = frase[corte + 1:].strip()
    if frase:
        salida.append(frase)
    return salida


# ── JavaScript que corre en el navegador ──
# Habla a través del `speechSynthesis` de la ventana PADRE de Streamlit: así la lectura no se corta
# cuando Streamlit recrea el iframe en cada actualización. Es idempotente: recibe SIEMPRE la lista
# completa de frases del turno y solo dice las que aún no ha dicho (V.n), de modo que si una
# actualización se pierde, la siguiente la recupera.
_JS_VOZ = r"""
(function () {
  var D = __DATA__, P, S;
  var estado = document.getElementById("estado"), boton = document.getElementById("detener");
  try { P = window.parent; S = P.speechSynthesis; void P.document; } catch (e) { S = null; }
  if (!S) {
    if (estado) { estado.textContent = "⚠️ Tu navegador no permite la lectura en voz alta. Prueba con Chrome, Edge o Safari."; }
    if (boton) { boton.disabled = true; }
    return;
  }
  var V = P.__vozTIC;
  if (!V) {
    V = P.__vozTIC = { turno: null, n: 0, frases: [], vivas: [], activa: null, cfg: D.cfg };
    // Algunos navegadores (iOS/Safari) solo dejan hablar tras un gesto del usuario: se «desbloquea» con el primero.
    ["pointerdown", "keydown"].forEach(function (ev) {
      P.document.addEventListener(ev, function f() {
        P.document.removeEventListener(ev, f, true);
        try { var u = new P.SpeechSynthesisUtterance(" "); u.volume = 0; S.speak(u); } catch (e) {}
      }, true);
    });
  }
  V.cfg = D.cfg;

  // Se reinstalan los métodos en cada carga para que pertenezcan siempre al iframe vivo.
  V.elegir = function () {
    var c = V.cfg, idiomas = c.idiomas.map(function (x) { return x.toLowerCase(); });
    var voces = S.getVoices().filter(function (v) { return /^es([-_]|$)/i.test(v.lang); });
    if (!voces.length) { return null; }
    function puntos(v) {
      var p = 0, nom = v.name.toLowerCase(), i = idiomas.indexOf(v.lang.replace("_", "-").toLowerCase());
      if (i >= 0) { p += (idiomas.length - i) * 10; }
      for (var j = 0; j < c.preferidas.length; j++) {
        if (nom.indexOf(c.preferidas[j].toLowerCase()) >= 0) { p += 50 + (c.preferidas.length - j); break; }
      }
      if (c.preferirLocal && v.localService) { p += 1000; }  // una voz instalada gana siempre: arranca al instante y el texto no sale del equipo
      return p;
    }
    voces.sort(function (a, b) { return puntos(b) - puntos(a); });
    return voces[0];
  };
  V.decir = function (texto) {
    var u = new P.SpeechSynthesisUtterance(texto), voz = V.elegir();
    if (voz) { u.voice = voz; u.lang = voz.lang; } else { u.lang = V.cfg.idioma; }
    u.rate = V.cfg.velocidad;
    u.pitch = V.cfg.tono;
    V.vivas.push(u);  // evita que Chrome recolecte la frase antes de terminar de hablarla
    u.onend = u.onerror = function () { var i = V.vivas.indexOf(u); if (i >= 0) { V.vivas.splice(i, 1); } };
    S.speak(u);
  };
  V.encolar = function (turno, frases) {
    if (V.turno !== turno) { S.cancel(); V.turno = turno; V.n = 0; }  // turno nuevo = se calla lo anterior
    V.frases = frases;
    while (V.n < V.frases.length) { V.decir(V.frases[V.n]); V.n++; }
  };
  V.detener = function () { V.turno = null; V.n = 0; V.frases = []; S.cancel(); };

  if (D.accion === "hablar") { V.encolar(D.turno, D.frases); }

  if (D.accion === "barra") {
    if (V.activa !== D.cfg.activa && !D.cfg.activa) { V.detener(); }  // al silenciar la voz se calla lo que sonaba
    V.activa = D.cfg.activa;
    var t0 = Date.now();
    var pintar = function () {
      var habla = S.speaking || S.pending, sinVoz = Date.now() - t0 > 2500 && !V.elegir();
      var msg = habla ? "🔊 Leyendo en voz alta…"
        : (D.cfg.activa ? "🔈 Voz lista: leeré cada respuesta" : "🔇 Voz desactivada (actívala en el menú lateral)");
      if (!habla && sinVoz) { msg += " · ⚠️ este equipo no tiene voz en español instalada"; }
      if (estado.textContent !== msg) { estado.textContent = msg; }
      boton.disabled = !habla;
    };
    boton.onclick = function () { V.detener(); pintar(); };
    pintar();
    setInterval(pintar, 300);
  }
})();
"""

_HTML_BARRA = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><style>
html,body{margin:0;font-family:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;}
body{padding:4px 0;}
.barra{display:flex;align-items:center;gap:12px;box-sizing:border-box;min-height:52px;padding:4px 12px;background:#fff;
 border:1px solid #D9DCEA;border-left:6px solid #6F42C1;border-radius:12px;color:#003366;font-size:15px;line-height:1.3;}
#estado{flex:1;}
button{min-height:44px;min-width:44px;padding:0 16px;border:0;border-radius:10px;background:#B42318;color:#fff;
 font-size:15px;font-weight:700;cursor:pointer;}
button:disabled{background:#E4E6F0;color:#4B5163;cursor:not-allowed;}
button:focus-visible{outline:3px solid #6F42C1;outline-offset:2px;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;}}
</style></head><body>
<div class="barra" role="region" aria-label="Control de la lectura en voz alta">
 <span id="estado" role="status" aria-live="polite">Cargando voz…</span>
 <button id="detener" type="button" disabled title="Detiene la lectura en voz alta" aria-label="Detener la lectura en voz alta">⏹ Detener voz</button>
</div>
<script>__JS__</script></body></html>"""


def _html_voz(datos: dict, barra: bool) -> str:
    carga = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
    js = _JS_VOZ.replace("__DATA__", carga)
    if barra:
        return _HTML_BARRA.replace("__JS__", js)
    return f'<!doctype html><html lang="es"><body><script>{js}</script></body></html>'


def _iframe(html: str, alto: int, titulo: str) -> None:
    """Incrusta HTML con JavaScript. Usa st.iframe (Streamlit reciente) o components.html (antiguo)."""
    alto = max(1, alto)  # st.iframe no admite altura 0; con 1 px y el CSS de arriba el iframe de voz no se ve
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=alto, alt=titulo)
        except TypeError:  # versiones sin el parámetro «alt»
            st.iframe(html, height=alto)
        return
    from streamlit.components.v1 import html as _html

    _html(html, height=alto)


def config_voz() -> dict:
    """Parámetros de voz = lo fijado en CONFIGURACIÓN + lo que la persona eligió en la barra lateral."""
    ss = st.session_state
    return dict(
        activa=bool(ss.get("voz_on", VOZ_INICIAL)),
        velocidad=VOZ_VELOCIDADES.get(ss.get("voz_vel", VOZ_VELOCIDAD_INICIAL), 1.0),
        tono=VOZ_TONO,
        idioma=VOZ_IDIOMA,
        idiomas=list(VOZ_IDIOMAS),
        preferidas=list(VOZ_PREFERIDAS),
        preferirLocal=VOZ_PREFERIR_LOCAL,
    )


def barra_voz(cfg: dict) -> None:
    """H1 · H3 · H11: estado de la voz en vivo y botón «Detener voz» siempre a la vista."""
    _iframe(_html_voz({"accion": "barra", "cfg": cfg}, barra=True), 68, "Control de voz")


def enviar_voz(destino, turno: str, frases: list[str], cfg: dict) -> None:
    """Manda al navegador la lista COMPLETA de frases del turno (solo dirá las nuevas). Turno nuevo = calla lo anterior."""
    with destino:
        _iframe(_html_voz({"accion": "hablar", "turno": turno, "frases": frases, "cfg": cfg}, barra=False), 0, "Voz del navegador")


class Locutor:
    """Convierte el texto que llega en streaming en frases completas y las lee apenas están listas."""

    def __init__(self, destino, cfg: dict):
        self.destino, self.cfg = destino, cfg
        self.turno = uuid.uuid4().hex
        self.buf, self.frases, self.en_codigo = "", [], False
        if destino is not None:
            enviar_voz(destino, self.turno, [], cfg)  # al empezar el turno se calla lo que estuviera sonando

    def _limpiar(self, frase: str) -> list[str]:
        if frase.lstrip().startswith("```"):  # los bloques de código no se leen
            self.en_codigo = not self.en_codigo
            return []
        if self.en_codigo:
            return []
        limpia = texto_para_voz(frase)
        return trocear(limpia) if limpia else []

    def _procesar(self, final: bool) -> None:
        nuevas, self.buf = extraer_frases(self.buf, final)
        antes = len(self.frases)
        for f in nuevas:
            self.frases += self._limpiar(f)
        if len(self.frases) > antes and self.destino is not None and self.cfg["activa"]:
            enviar_voz(self.destino, self.turno, self.frases, self.cfg)

    def alimentar(self, trozo: str) -> None:
        self.buf += trozo
        self._procesar(final=False)

    def cerrar(self) -> None:
        self._procesar(final=True)


def frases_de_texto(texto: str, pii: bool) -> list[str]:
    """Frases listas para leer de un mensaje ya escrito (botón «Escuchar»)."""
    locutor = Locutor(None, {"activa": True})
    locutor.alimentar(enmascarar_pii(texto)[0] if pii else texto)
    locutor.cerrar()
    return locutor.frases


def boton_escuchar(idx: int, texto: str, ph_voz, pii: bool, cfg: dict) -> None:
    """H4: el mismo botón «Escuchar» en todas las respuestas (H7: sirve aunque la lectura automática esté apagada)."""
    if st.button("🔊 Escuchar", key=f"voz_{idx}", help="Lee esta respuesta en voz alta con la voz de tu navegador."):
        frases = frases_de_texto(texto, pii)
        if frases:
            enviar_voz(ph_voz, uuid.uuid4().hex, frases, cfg)


# ───────────────────────── Datos de contenido ─────────────────────────
EJEMPLOS = [
    "¿Qué dice Reamer sobre la privacidad digital?",
    "¿Cuáles son los 3 niveles de la brecha digital?",
    "¿Cómo evitar la deshumanización al usar IA?",
]

HERRAMIENTAS = {
    "🗂️ Gestión de Casos": [
        dict(icono="🗂️", nombre="Airtable", tipo="Base de datos relacional en la nube (afín: Microsoft Access)",
             ventajas=["Relaciona personas, familias e intervenciones", "Vistas por estado de caso y filtros", "Permisos de acceso por rol"],
             uso="Expediente digital único: la persona no repite su historia ante cada profesional (evita la revictimización).",
             cuidado="Activar control de acceso y no cargar datos sensibles sin consentimiento informado."),
        dict(icono="📌", nombre="Trello", tipo="Tableros Kanban",
             ventajas=["Tablero visual: por hacer / en curso / cerrado", "Fechas límite y recordatorios", "Etiquetas de prioridad"],
             uso="Seguimiento del plan de intervención y de las visitas domiciliarias programadas.",
             cuidado="Usar códigos o iniciales, no nombres completos, en las tarjetas."),
        dict(icono="✅", nombre="Asana", tipo="Gestión de proyectos y tareas",
             ventajas=["Tareas con responsables y subtareas", "Línea de tiempo del proyecto", "Coordinación de equipos multidisciplinarios"],
             uso="Planificar programas comunitarios y repartir tareas entre trabajo social, salud y educación.",
             cuidado="Limitar quién ve cada proyecto; no adjuntar informes confidenciales."),
    ],
    "🔍 Investigación y Diagnóstico": [
        dict(icono="📱", nombre="KoboToolbox", tipo="Encuestas móviles offline",
             ventajas=["Funciona sin Internet y sincroniza después", "Recolección en campo con celular o tablet", "Exporta datos a Excel/CSV"],
             uso="Fichas socioeconómicas y padrones en comunidades rurales del Altiplano con poca conectividad.",
             cuidado="Proteger el dispositivo con clave e incluir el consentimiento informado al inicio del formulario."),
        dict(icono="📊", nombre="SPSS", tipo="Análisis cuantitativo",
             ventajas=["Tabulación y cruce de variables", "Estadística descriptiva", "Gráficos para informes"],
             uso="Procesar diagnósticos sociales (acceso a servicios, violencia, pobreza) y sustentar proyectos con evidencia.",
             cuidado="Trabajar con bases anonimizadas."),
        dict(icono="🧠", nombre="ATLAS.ti", tipo="Análisis cualitativo (CAQDAS)",
             ventajas=["Codifica entrevistas y diarios de campo", "Organiza grupos focales", "Visualiza relaciones entre categorías"],
             uso="Identificar necesidades subyacentes y discursos de los usuarios.",
             cuidado="Anonimizar transcripciones y resguardar los audios originales."),
    ],
    "📣 Promoción y Comunicación": [
        dict(icono="🎨", nombre="Canva", tipo="Diseño gráfico intuitivo",
             ventajas=["Plantillas de afiches, infografías y pósteres", "No requiere diseñador", "La comunidad crea su propio mensaje"],
             uso="Campañas de prevención y materiales educativos; fortalece las habilidades digitales.",
             cuidado="Pedir autorización para usar fotografías de personas, sobre todo de menores."),
        dict(icono="🎥", nombre="Zoom / Google Meet", tipo="Videollamadas",
             ventajas=["Entrevistas y reuniones a distancia", "Coordinación interinstitucional", "Útil ante problemas de movilidad"],
             uso="Orientación a familias alejadas de los servicios sociales en comunidades rurales de Puno (si hay conexión).",
             cuidado="Verificar un espacio privado, consentimiento informado y capacidades digitales de la persona."),
        dict(icono="📱", nombre="Redes sociales (Facebook, TikTok)", tipo="Redes de articulación y audiovisuales",
             ventajas=["Interacción bidireccional con la comunidad", "Narrativas cortas que llegan a jóvenes", "Convocatoria y redes de apoyo"],
             uso="Campañas sobre prevención de la violencia de género, diversidad familiar y derechos humanos.",
             cuidado="Nunca exponer casos individuales; cuidar los límites profesionales en perfiles personales."),
    ],
    "🤖 Empoderamiento e IA": [
        dict(icono="🤖", nombre="Asistentes conversacionales", tipo="Inteligencia artificial generativa",
             ventajas=["Orientación y resúmenes de normativa", "Explican trámites en lenguaje sencillo", "Apoyan la gestión del conocimiento"],
             uso="Preparar material educativo y orientar rutas de atención, como apoyo y nunca como sustituto del criterio profesional.",
             cuidado="No ingresar datos identificables y verificar sus respuestas (riesgo de errores y deshumanización)."),
        dict(icono="🌎", nombre="Traductores automáticos", tipo="Quechua ↔ Español (p. ej. DeepL, Google Traductor)",
             ventajas=["Reducen barreras de idioma", "Traducen avisos y formularios rápidamente", "Facilitan el primer contacto"],
             uso="Materiales informativos para población quechua y aimara hablante del Altiplano.",
             cuidado="La calidad varía según el idioma: validar con intérpretes o hablantes nativos."),
        dict(icono="♿", nombre="Accesibilidad cognitiva", tipo="Lectura fácil, pictogramas y audio",
             ventajas=["Simplifican textos y trámites", "Lectura en voz alta y apoyos visuales", "Útiles para personas mayores o con discapacidad"],
             uso="Adaptar consentimientos informados para que la persona comprenda y decida con autonomía.",
             cuidado="La simplificación no debe omitir información relevante."),
    ],
}

NIVELES = [
    dict(n=1, color=AZUL, titulo="Acceso físico", autor="van Dijk (2006) · Gómez Navarro et al. (2018)",
         desc="Diferencias en dispositivos, infraestructura y conexión a Internet, incluida la brecha urbano-rural y la capacidad de pagar equipos y servicios.",
         ts="Identificar barreras de acceso y buscar alternativas (atención presencial, puntos de conectividad)."),
    dict(n=2, color=PURPURA, titulo="Competencias digitales", autor="van Dijk (2006)",
         desc="Habilidades para usar la tecnología: alfabetización digital, edad, nivel educativo y condiciones sociales.",
         ts="Acompañar los trámites en línea y facilitar talleres de alfabetización digital."),
    dict(n=3, color="#0F766E", titulo="Exclusión y apropiación social", autor="Gómez Navarro et al. (2018)",
         desc="Tener acceso no garantiza aprovecharlo: la desigualdad en uso y apropiación limita la participación social de ciertos grupos.",
         ts="Promover inclusión y ciudadanía digital, y abogar para que la tecnología no sea una nueva exclusión."),
]

DIAGNOSTICO = {
    1: ["Dispone de celular, tablet o computadora que puede usar con regularidad", "Tiene señal o Internet estable donde vive y puede costearlo"],
    2: ["Sabe usar aplicaciones básicas (mensajería, formularios)", "Puede buscar información y hacer trámites en línea con seguridad"],
    3: ["Usa la tecnología para sus propios fines (derechos, ingresos, organización)", "Participa en redes o espacios digitales de su comunidad"],
}

DEBATE_SI = {
    "🧾 Automatizar la burocracia": "Informes, formatos, citas y registros se hacen en una fracción del tiempo, liberando horas para la relación de ayuda (Santas García, 2016).",
    "🗃️ Organizar los datos": "Un expediente digital único y trazable permite el trabajo en red y evita que la persona repita su historia traumática (revictimización).",
    "📡 Teleasistencia": "Seguimiento preventivo a personas mayores y a familias alejadas, cuando la visita domiciliaria no es posible y existe conectividad (Gutiérrez Campos, 2025).",
}
DEBATE_NO = {
    "💞 Empatía": "Ninguna herramienta comprende el sufrimiento ni construye confianza: es la base de la relación profesional (Reamer, 2014).",
    "👂 Escucha activa": "La comunicación presencial permite captar emociones y circunstancias que un sistema reduce a datos (Nordesjö y Scaramuzzino, 2023).",
    "⚖️ Juicio ético": "Decidir sobre derechos o medidas de protección no se delega a algoritmos: existe el riesgo de deshumanización (Hwang, 2024).",
    "🤝 Acompañamiento humano": "La visita, la presencia en crisis y la defensa de derechos son humanas; la tecnología solo las amplía.",
}

CASOS = [
    dict(
        titulo="Caso 1 · El informe por WhatsApp personal",
        situacion="Una trabajadora social recibe el informe de un niño en situación de riesgo. Para «agilizar», quiere reenviarlo completo desde su "
                  "WhatsApp personal al director de la institución educativa.",
        opciones=[
            dict(texto="Reenviar el informe completo por WhatsApp personal: es lo más rápido.", tipo="bad",
                 feedback="Riesgo alto: pérdida del dispositivo, acceso no autorizado y difusión sin consentimiento. Las normas de NASW et al. (2017) piden evitar plataformas inseguras."),
            dict(texto="Compartir solo la información mínima necesaria, por un canal institucional seguro y con base de consentimiento.", tipo="ok",
                 feedback="Correcto: equilibra la coordinación para proteger al niño con la confidencialidad y la seguridad de la información (Reamer, 2017; NASW et al., 2017)."),
            dict(texto="No compartir nada ni coordinar con nadie.", tipo="warn",
                 feedback="Evita el riesgo digital, pero puede dejar sin respuesta al niño. La ética exige equilibrar confidencialidad y protección, no renunciar a la coordinación."),
        ],
        preguntas=["¿Qué datos eran estrictamente necesarios para el director?", "¿Qué canal institucional seguro existe en su servicio?", "¿Cómo y cuándo se informa a la familia?"],
    ),
    dict(
        titulo="Caso 2 · La app que dejó fuera a doña Basilia",
        situacion="Una municipalidad decide que un programa de apoyo solo se solicitará mediante una app con chatbot que clasifica automáticamente a los postulantes. "
                  "Doña Basilia, de 78 años, aimara hablante y sin smartphone ni señal estable, queda sin registrarse.",
        opciones=[
            dict(texto="Aplicar la regla: si no usa la app, no accede. La tecnología es neutral.", tipo="bad",
                 feedback="Convierte la brecha digital en exclusión social (Gómez Navarro et al., 2018; van Dijk, 2006)."),
            dict(texto="Acompañarla en el registro con su consentimiento, explicarle en su lengua y abogar por una vía de atención no digital.", tipo="ok",
                 feedback="Correcto: respeta su autonomía, cierra la brecha en los tres niveles y trata el acceso como derecho (enfoque de ciudadanía digital)."),
            dict(texto="Registrarla por su cuenta, sin explicarle ni pedir su consentimiento, para que no pierda el beneficio.", tipo="warn",
                 feedback="Resuelve el trámite, pero vulnera su autonomía y el consentimiento informado (FITS, 2018), y no cambia la barrera estructural."),
        ],
        preguntas=["¿Qué nivel de brecha digital se observa en este caso?", "¿Qué decisiones no debería tomar un algoritmo por sí solo?", "¿Qué rol tiene el trabajador social frente a la política municipal?"],
    ),
]


# ───────────────────────── Componentes de interfaz ─────────────────────────
def mostrar_logo() -> None:
    for nombre in ("escudo_unap.png", "escudo_unap.jpg", "escudo_unap.webp"):
        archivo = BASE_DIR / nombre
        if archivo.is_file():
            st.image(str(archivo), width=120)
            return
    if LOGO_URL:
        st.image(LOGO_URL, width=120)
        return
    st.markdown('<div class="escudo">UNA<br>PUNO</div>', unsafe_allow_html=True)


def barra_lateral() -> tuple[bool, str]:
    """H8: solo controles útiles para la persona usuaria; la configuración técnica vive en el código."""
    with st.sidebar:
        mostrar_logo()
        st.markdown(f"<h3 style='text-align:center;color:{AZUL};margin:0'>{FACULTAD}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Curso:** {CURSO}")
        st.caption(DOCENTE)
        st.divider()
        pii = st.toggle(
            "🔒 Filtro Ético de Privacidad (Ley N.° 29733)",
            value=FILTRO_PII_INICIAL,
            help="Oculta DNI (8 dígitos) y teléfonos (9 dígitos) en tu consulta, en la respuesta y en la voz.",
        )
        st.toggle(
            "🔊 Leer respuestas en voz alta",
            value=VOZ_INICIAL,
            key="voz_on",
            help="Usa la voz de tu navegador: empieza a hablar mientras la respuesta se escribe.",
        )
        st.select_slider(
            "Velocidad de la voz",
            options=list(VOZ_VELOCIDADES),
            value=VOZ_VELOCIDAD_INICIAL,
            key="voz_vel",
            help="Se aplica a las frases que se lean a partir de ahora.",
        )
        st.divider()
        contexto, estado, ok = obtener_contexto()
        (st.success if ok else st.warning)(estado)  # H1
        st.markdown(
            "<div class='pie'>© 2026 · Escuela Profesional de Trabajo Social<br>Universidad Nacional del Altiplano - Puno</div>",
            unsafe_allow_html=True,
        )
    return pii, contexto


# ── Asistente: estado y callbacks ──
def _saludo() -> dict:
    return {
        "role": "assistant",
        "content": "¡Hola! Soy el asistente de protocolos TIC. Pregúntame sobre herramientas digitales, ética, brecha digital o cómo humanizar la intervención. Puedes escribir o grabar tu pregunta 🎙️.",
    }


def _borrar_conversacion() -> None:
    ss = st.session_state
    ss.messages = [_saludo()]
    ss.pop("fallida", None)
    ss["detener_voz"] = True


def _marcar_reintento() -> None:
    st.session_state["reintentar"] = True


def pintar_historial(ph_voz, pii: bool, cfg: dict) -> None:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("nota"):
                st.caption(m["nota"])
            if m["role"] == "assistant":
                boton_escuchar(i, m["content"], ph_voz, pii, cfg)


def pintar_fallo(f: dict) -> None:
    """H9: el error dice qué pasó y qué hacer, no se pierde la pregunta y se puede reintentar con un clic."""
    with st.chat_message("user"):
        st.markdown(f["resumen"])
    with st.chat_message("assistant"):
        st.error(f["mensaje"])
        with st.expander("🔧 Detalles técnicos (para quien administra la aplicación)"):
            st.code(f["detalle"], language=None)
        st.button("🔄 Reintentar", key="btn_reintentar", on_click=_marcar_reintento, type="primary")


def _emitir(flujo, locutor: Locutor, acum: dict, ph_estado):
    """Reparte cada trozo de texto: pantalla (write_stream) + voz (frases completas)."""
    primero = True
    try:
        for trozo in flujo:
            if primero:
                ph_estado.empty()  # H1: el aviso «Consultando…» se va al llegar el primer texto
                primero = False
            acum["texto"] += trozo
            locutor.alimentar(trozo)
            yield trozo
    finally:
        locutor.cerrar()  # lee la última frase, aunque no termine en punto o haya habido un corte


def responder(consulta: dict, contexto: str, api_key: str, pii: bool, ph_voz, cfg: dict) -> None:
    """Un turno completo: burbuja de la persona → respuesta en streaming con voz → historial."""
    ss = st.session_state
    sistema = construir_sistema(contexto, pii)
    previos = ss.messages[-HISTORIAL_MENSAJES:]
    while previos and previos[0]["role"] != "user":
        previos = previos[1:]
    historial = [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])])
        for m in previos
    ]

    es_audio = "audio" in consulta
    nota = None
    if es_audio:
        orden = (
            "Escucha la pregunta grabada y respóndela. Usa EXACTAMENTE este formato:\n"
            "TRANSCRIPCIÓN: <lo que dijo la persona, textual>\nRESPUESTA: <tu respuesta>"
        )
        if pii:
            orden += "\nSi la persona dicta DNI (8 dígitos) o teléfonos (9 dígitos), escríbelos como [DNI_OCULTO] o [TELÉFONO_OCULTO]."
        partes = [types.Part.from_bytes(data=consulta["audio"], mime_type="audio/wav"), types.Part.from_text(text=orden)]
        contenido = historial + [types.Content(role="user", parts=partes)]
        resumen = "🎤 *Pregunta por voz*"
    else:
        pregunta, n_dni, n_tel = enmascarar_pii(consulta["texto"]) if pii else (consulta["texto"], 0, 0)
        if n_dni or n_tel:
            nota = f"🔒 Filtro de privacidad: {n_dni} DNI y {n_tel} teléfono(s) ocultados antes de enviar."
        contenido = historial + [types.Content(role="user", parts=[types.Part.from_text(text=pregunta)])]
        resumen = pregunta

    # H1: la pregunta aparece al instante, antes de que el modelo responda
    with st.chat_message("user"):
        ph_usuario = st.empty()
        ph_usuario.markdown("🎤 *Transcribiendo tu pregunta…*" if es_audio else resumen)
        if nota:
            st.caption(nota)

    transcripcion: dict = {"texto": None}

    def al_transcribir(t: str) -> None:
        t = enmascarar_pii(t)[0] if pii else t
        transcripcion["texto"] = t
        ph_usuario.markdown(f"🎤 *Pregunta por voz:* «{t}»")

    ss.pop("modelo_usado", None)
    acum = {"texto": ""}
    locutor = Locutor(ph_voz, cfg)  # calla la voz anterior y queda listo para hablar frase por frase
    with st.chat_message("assistant"):
        ph_estado = st.empty()
        ph_estado.caption("⏳ Consultando el documento base…")  # H1
        flujo = flujo_gemini(api_key, contenido, sistema)
        if pii:
            flujo = filtrar_pii_stream(flujo)
        if es_audio:
            flujo = separar_transcripcion(flujo, al_transcribir)
        error: Exception | None = None
        try:
            st.write_stream(_emitir(flujo, locutor, acum, ph_estado))
        except Exception as e:
            error = e
        ph_estado.empty()

        respuesta = acum["texto"].strip()
        if not respuesta and error is None:
            error = RuntimeError("La respuesta llegó vacía.")
        if error is not None:
            log.warning("Fallo en la consulta: %r", error)
            if not respuesta:  # no se mostró nada: se guarda la consulta para «Reintentar»
                ss["fallida"] = dict(consulta=consulta, resumen=resumen, mensaje=mensaje_error(error), detalle=f"{type(error).__name__}: {error}")
                st.rerun()
        aviso = f"⚠️ La respuesta se interrumpió antes de terminar. {mensaje_error(error)}" if error is not None else None

        usado = ss.get("modelo_usado", MODELO)
        nota_ia = f"ℹ️ Respondió {usado}: el modelo principal estaba saturado o no disponible." if usado != MODELO else None
        nota_ia = aviso or nota_ia
        if nota_ia:
            st.caption(nota_ia)

        msg_usuario = f"🎤 *Pregunta por voz:* «{transcripcion['texto'] or '(no disponible)'}»" if es_audio else resumen
        ss.messages.append({"role": "user", "content": msg_usuario, "nota": nota})
        ss.messages.append({"role": "assistant", "content": respuesta, "nota": nota_ia})
        ss.pop("fallida", None)
        boton_escuchar(len(ss.messages) - 1, respuesta, ph_voz, pii, cfg)  # H4: igual que en el historial


def tab_asistente(contexto: str, api_key: str, pii: bool, cfg: dict) -> None:
    ss = st.session_state
    listo = GENAI_OK and bool(api_key)

    cab, borrar = st.columns([4, 1], vertical_alignment="center")
    cab.subheader("🤖 Asistente de Protocolos TIC")
    with borrar.popover("🗑️ Nueva conversación"):  # H3 + H5: se puede empezar de cero, con confirmación
        st.write("Se borrará el historial de esta sesión.")
        st.button("Sí, borrar", key="btn_borrar", type="primary", on_click=_borrar_conversacion)
    st.caption("Responde con base en la monografía del curso (Reamer, Castells, van Dijk, NASW/FITS). La respuesta se lee en voz alta mientras se escribe.")
    with st.expander("ℹ️ ¿Cómo usar este asistente?"):  # H10
        st.markdown(
            "1. **Escribe, graba o elige** una pregunta rápida.\n"
            "2. La respuesta **aparece mientras se genera** y se **lee en voz alta** con la voz de tu navegador.\n"
            "3. **⏹ Detener voz** calla la lectura; **🔊 Escuchar** repite cualquier respuesta.\n"
            "4. Cambia la velocidad o apaga la voz en el menú lateral.\n"
            "5. **No escribas ni dictes datos reales de casos.** El filtro oculta DNI y teléfonos, pero no nombres ni direcciones.\n\n"
            "*La calidad de la voz depende de tu dispositivo; suele sonar mejor en Chrome o Edge.*"
        )

    if not GENAI_OK:  # H9: qué pasa y a quién acudir, sin jerga
        st.error("🔌 El asistente no está disponible: falta instalar un componente en el servidor. Avisa a quien administra la aplicación.")
    elif not api_key:
        st.error("🔑 El asistente aún no está activado: falta la clave del servicio de IA en la sección CONFIGURACIÓN de app.py.")
    if not contexto:
        st.warning("📄 Sin documento base: el asistente responderá solo con conocimiento general.")

    barra_voz(cfg)  # H1 · H3 · H11
    ph_voz = st.empty()  # aquí se inyecta (sin verse) el script que habla
    if ss.pop("detener_voz", False):
        enviar_voz(ph_voz, uuid.uuid4().hex, [], cfg)
    zona = st.container()  # historial + respuesta en curso: queda ARRIBA de los controles de entrada

    st.markdown("**💡 Preguntas rápidas**")  # H6
    for col, pregunta in zip(st.columns(len(EJEMPLOS)), EJEMPLOS):
        if col.button(pregunta, key=f"ej_{pregunta}", disabled=not listo):
            ss.pregunta_rapida = pregunta

    audio = None
    if listo and hasattr(st, "audio_input"):
        st.markdown("**🎙️ Consulta por voz**")
        audio = st.audio_input("Graba tu pregunta", key="audio_pregunta", label_visibility="collapsed")
        if pii:
            st.caption("🔒 El audio no puede filtrarse antes de enviarse: no dictes datos reales. La transcripción y la respuesta sí se filtran.")
    texto = st.chat_input("Escribe tu consulta sobre TIC en Trabajo Social…", max_chars=MAX_CONSULTA, disabled=not listo)  # H5
    texto = texto or ss.pop("pregunta_rapida", None)

    # ¿Qué hay que responder en esta ejecución?
    nueva: dict | None = None
    if texto:
        nueva = {"texto": texto}
    elif audio is not None:
        datos = audio.getvalue()
        huella = hashlib.md5(datos).hexdigest()
        if huella != ss.get("audio_visto"):  # la misma grabación no se procesa dos veces
            ss["audio_visto"] = huella
            nueva = {"audio": datos}
    if ss.pop("reintentar", False) and ss.get("fallida"):
        nueva = ss["fallida"]["consulta"]
    if nueva:
        ss.pop("fallida", None)

    with zona:
        pintar_historial(ph_voz, pii, cfg)
        if nueva and listo:
            responder(nueva, contexto, api_key, pii, ph_voz, cfg)
        elif ss.get("fallida"):
            pintar_fallo(ss["fallida"])


def tarjeta_herramienta(h: dict) -> None:
    with st.container(border=True):
        st.markdown(f"### {h['icono']} {h['nombre']}")
        st.caption(h["tipo"])
        st.markdown("**✅ Ventajas**\n" + "\n".join(f"- {v}" for v in h["ventajas"]))
        st.markdown(f"**🧩 En Trabajo Social:** {h['uso']}")
        st.markdown(f"⚠️ *Cuidado ético:* {h['cuidado']}")


def tab_matriz() -> None:
    st.subheader("📊 Matriz de Herramientas TIC")
    st.caption("Filtra por categoría y revisa ventajas, aplicación en Trabajo Social y cuidados éticos.")
    opciones = ["Todas"] + list(HERRAMIENTAS)
    eleccion = st.radio("Categoría", opciones, horizontal=True, key="cat_matriz")
    for categoria, lista in HERRAMIENTAS.items():
        if eleccion not in ("Todas", categoria):
            continue
        st.markdown(f"### {categoria}")
        for col, herramienta in zip(st.columns(3), lista):
            with col:
                tarjeta_herramienta(herramienta)


def tab_laboratorio() -> None:
    st.subheader("🛡️ Laboratorio de Ética y Brecha Digital")
    st.markdown("### 🔐 1. Prueba de privacidad: anonimización de fichas")
    izq, der = st.columns([3, 2])
    with izq:
        ficha = st.text_area(
            "Ficha de atención (usa datos ficticios)",
            value="Atención a la usuaria María Pérez DNI 71234567 tel 987654321",
            height=140,
            key="ficha_pii",
        )
        if st.button("🔒 Anonimizar ficha", type="primary"):
            st.session_state.resultado_pii = enmascarar_pii(ficha)
        resultado = st.session_state.get("resultado_pii")
        if resultado:
            anonimo, n_dni, n_tel = resultado
            st.success(anonimo.replace("_", "\\_"))
            c1, c2 = st.columns(2)
            c1.metric("DNI ocultos", n_dni)
            c2.metric("Teléfonos ocultos", n_tel)
            st.caption("⚠️ El filtro detecta solo patrones numéricos: nombres y direcciones no se ocultan (observa «María Pérez»). Por eso hay que minimizar datos y usar canales seguros.")
    with der:
        st.info(
            "**Ley N.° 29733 · Protección de Datos Personales (Perú)**\n\n"
            "- Protege los datos que identifican a una persona: DNI, teléfono, dirección, imagen, voz.\n"
            "- Principios clave: consentimiento, finalidad, proporcionalidad, calidad y seguridad.\n"
            "- Los datos sensibles (salud, violencia, menores en riesgo) exigen mayor protección y consentimiento reforzado.\n"
            "- La persona puede ejercer sus derechos de acceso, rectificación, cancelación y oposición.\n"
            "- En Trabajo Social se conecta con confidencialidad y consentimiento informado (FITS, 2018; NASW et al., 2017).\n\n"
            "*Contenido de apoyo: verifica el texto y su reglamento vigentes antes de citarlos.*"
        )

    st.divider()
    st.markdown("### 🌐 2. Los 3 niveles de la brecha digital")
    for col, nv in zip(st.columns(3), NIVELES):
        col.markdown(
            f'<div class="nivel" style="--c:{nv["color"]}"><span class="badge">{nv["n"]}</span><b>{nv["titulo"]}</b><br>'
            f'<small>{nv["autor"]}</small><p>{nv["desc"]}</p><p><b>🧭 Rol del trabajador social:</b> {nv["ts"]}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### 🩺 Mini-diagnóstico: ¿dónde está la barrera de una persona o comunidad?")
    marcados = {}
    for col, (nivel, items) in zip(st.columns(3), DIAGNOSTICO.items()):
        with col:
            st.markdown(f"**Nivel {nivel} · {NIVELES[nivel - 1]['titulo']}**")
            marcados[nivel] = sum(st.checkbox(item, key=f"diag_{nivel}_{i}") for i, item in enumerate(items))
    total, maximo = sum(marcados.values()), sum(len(v) for v in DIAGNOSTICO.values())
    st.progress(total / maximo, text=f"Condiciones cumplidas: {total} de {maximo}")
    prioridad = next((n for n, items in DIAGNOSTICO.items() if marcados[n] < len(items)), None)
    if prioridad is None:
        st.success("✅ Sin barreras en los tres niveles: la persona o comunidad puede participar plenamente en entornos digitales.")
    else:
        st.warning(f"🎯 Nivel prioritario de intervención: **{prioridad} · {NIVELES[prioridad - 1]['titulo']}**. {NIVELES[prioridad - 1]['ts']}")


def tab_debate() -> None:
    st.subheader("⚖️ Debate: Humanización vs. Automatización")
    st.caption("Abre cada tarjeta para ver el fundamento. Luego resuelve los casos éticos con el público.")
    izq, der = st.columns(2)
    with izq:
        st.markdown('<div class="col-si">✅ Lo que la TIC SÍ debe hacer</div>', unsafe_allow_html=True)
        for titulo, texto in DEBATE_SI.items():
            with st.expander(titulo):
                st.write(texto)
    with der:
        st.markdown('<div class="col-no">⛔ Lo que la TIC NUNCA debe reemplazar</div>', unsafe_allow_html=True)
        for titulo, texto in DEBATE_NO.items():
            with st.expander(titulo):
                st.write(texto)
    st.info("💡 Idea fuerza: las TIC automatizan la burocracia para liberar tiempo y que el profesional pueda escuchar, acompañar y resolver los problemas reales de las personas.")

    st.divider()
    st.markdown("### 🧪 Simulador de casos éticos")
    for idx, caso in enumerate(CASOS):
        with st.container(border=True):
            st.markdown(f"#### {caso['titulo']}")
            st.write(caso["situacion"])
            eleccion = st.radio("¿Qué decisión tomarías?", [o["texto"] for o in caso["opciones"]], index=None, key=f"caso_{idx}")
            if eleccion:
                opcion = next(o for o in caso["opciones"] if o["texto"] == eleccion)
                {"ok": st.success, "warn": st.warning, "bad": st.error}[opcion["tipo"]](opcion["feedback"])
            with st.expander("💭 Preguntas de reflexión para el jurado o el público"):
                for pregunta in caso["preguntas"]:
                    st.markdown(f"- {pregunta}")


# ───────────────────────── Aplicación ─────────────────────────
def main() -> None:
    st.session_state.setdefault("messages", [_saludo()])
    pii, contexto = barra_lateral()
    api_key = clave_api() if GENAI_OK else ""
    cfg = config_voz()

    st.markdown(
        '<div class="hero"><h1>Herramientas TIC para la Intervención en Trabajo Social</h1>'
        "<p>Tecnología ética, accesible e inclusiva: IA, voz y gestión de datos al servicio de la labor social, "
        "con confidencialidad y atención a la brecha digital.</p></div>",
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.tabs(["🤖 Asistente de Protocolos", "📊 Matriz de Herramientas", "🛡️ Ética y Brecha Digital", "⚖️ Debate"])
    with t1:
        tab_asistente(contexto, api_key, pii, cfg)
    with t2:
        tab_matriz()
    with t3:
        tab_laboratorio()
    with t4:
        tab_debate()


main()
