# -*- coding: utf-8 -*-
"""
Asistente de Protocolos TIC · Trabajo Social · UNA Puno   (solo el chatbot)

Ejecutar:    streamlit run app.py
Requisitos:  pip install -U "streamlit>=1.40" google-genai pypdf edge-tts gTTS
Documento:   coloca HERRAMIENTAS_TIC_TRABAJO_SOCIAL.pdf junto a este archivo.
Clave:       .streamlit/secrets.toml →  GEMINI_API_KEY = "tu-clave"
             (o variable de entorno GEMINI_API_KEY). No la pegues en el código.

LECTURA EN VOZ ALTA (texto → voz) · compatible con cualquier navegador
  El audio se genera en el SERVIDOR con una librería de Python (edge-tts; gTTS como respaldo) y se
  entrega al navegador como MP3 dentro de un reproductor estándar (st.audio). No depende de las voces
  instaladas en el equipo ni de la Web Speech API, que cambia mucho de un navegador a otro.
  · Al terminar cada respuesta se reproduce sola (si el navegador lo permite) y siempre queda el
    reproductor visible con ▶ por si el navegador bloquea el audio automático (p. ej. Safari/iOS).
  · «🔊 Escuchar» repite cualquier respuesta; «⏹ Detener voz» corta el audio.

CONSULTA POR MICRÓFONO (voz → texto) · nunca inventa la pregunta
  1. Compuerta local: si el audio es muy corto o casi no tiene voz (silencio/ruido), NI SIQUIERA se
     envía al modelo y se pide repetir. Es lo que evita que el modelo «alucine» una pregunta.
  2. Transcripción literal: el modelo solo transcribe (temperatura 0, salida JSON) e informa si se
     entiende y con qué confianza. No responde nada en este paso.
  3. Decisión:  confianza alta → responde · confianza media → pide confirmar lo que entendió ·
     baja / ininteligible / sospechosa → pide repetir el audio.
  4. Solo después de aceptar la transcripción se hace la consulta al chatbot, como texto normal.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

try:
    from google import genai
    from google.genai import types

    GENAI_OK = True
except Exception:  # SDK no instalado: la app avisa en pantalla
    GENAI_OK = False

try:
    import edge_tts
except Exception:
    edge_tts = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None
VOZ_OK = edge_tts is not None or gTTS is not None

log = logging.getLogger("tic_trabajo_social")

# ═══════════════════════════ CONFIGURACIÓN (BACKEND) ═══════════════════════════
# Lo técnico se cambia SOLO aquí. Nada de esto aparece en la interfaz.

# ── Servicio de IA (Gemini) ──
GEMINI_API_KEY = ""  # Déjala vacía: se lee de st.secrets["GEMINI_API_KEY"] o de la variable de entorno.
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Se prueban en orden si el modelo elegido está saturado (503/429) o ya no existe (404).
MODELOS_RESPALDO = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")
REINTENTOS_POR_MODELO = 2
PAUSA_REINTENTO_S = 1.5
TEMPERATURA: float | None = None  # None = valor por defecto del modelo (solo para el chat)
HISTORIAL_MENSAJES = 6  # mensajes previos que se reenvían al modelo como contexto
MAX_CONSULTA = 1500  # caracteres máximos por pregunta escrita

# ── Documento base ──
PDF_NOMBRE = "HERRAMIENTAS_TIC_TRABAJO_SOCIAL.pdf"
MAX_CONTEXTO = 120_000  # caracteres del PDF enviados al modelo
PERMITIR_SUBIR_PDF = False  # True = si falta el PDF local, la barra lateral deja subirlo

# ── Privacidad ──
FILTRO_PII_INICIAL = True  # el filtro (DNI y teléfonos) arranca activado

# ── Voz de salida (texto → voz) ──
VOZ_INICIAL = True  # lectura automática activada al abrir la app
VOCES = {  # nombre en pantalla → voz neuronal de Microsoft Edge (edge-tts)
    "Alex · hombre (Perú)": "es-PE-AlexNeural",
    "Camila · mujer (Perú)": "es-PE-CamilaNeural",
    "Jorge · hombre (México)": "es-MX-JorgeNeural",
    "Dalia · mujer (México)": "es-MX-DaliaNeural",
    "Álvaro · hombre (España)": "es-ES-AlvaroNeural",
    "Elvira · mujer (España)": "es-ES-ElviraNeural",
}
VOZ_POR_DEFECTO = "Alex · hombre (Perú)"
VOZ_VELOCIDADES = {"Lenta": 0.85, "Normal": 1.0, "Rápida": 1.25}
VOZ_VELOCIDAD_INICIAL = "Normal"
GTTS_TLD = "com.mx"  # acento del respaldo gTTS (com.mx = latinoamericano, es = España)
MAX_CHARS_VOZ = 3500  # una respuesta más larga se lee solo hasta aquí (cortando en una frase completa)
TTS_TIMEOUT_S = 45
# Cómo debe sonar lo que la voz suele leer mal (expresión regular → texto leído).
PRONUNCIACION = {
    r"\bet al\.": "y colaboradores",
    r"\bN\.\s?[°º]": "número",
    r"\bNASW\b": "N.A.S.W.",
    r"\bTICs?\b": "tics",
    r"\bIA\b": "inteligencia artificial",
    r"\bDNI\b": "D N I",
}

# ── Micrófono (voz → texto) ──
AUDIO_MIN_S = 0.8  # menos que esto: demasiado corto para ser una pregunta
AUDIO_MAX_S = 120.0
AUDIO_VOZ_MIN_S = 0.5  # segundos mínimos con señal de voz; menos = silencio o solo ruido
AUDIO_UMBRAL_RMS = 0.008  # volumen mínimo (0–1) de un tramo de 50 ms para contarlo como «con voz»
CONFIRMAR_SI_DUDA = True  # confianza «media» → pedir confirmación; False = responder directamente

# ── Identidad visual y textos institucionales ──
AZUL, PURPURA = "#003366", "#6F42C1"
LOGO_URL = ""  # URL del escudo de la UNA Puno (o guarda escudo_unap.png junto a app.py)
FACULTAD = "Facultad de Trabajo Social - UNA Puno"
CURSO = "TICs para la Intervención en Trabajo Social"
DOCENTE = "Docente: Sucapuca Tito, Elizabeth · VII semestre · 2026-II"
# ═════════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

EJEMPLOS = [
    "¿Qué dice Reamer sobre la privacidad digital?",
    "¿Cuáles son los 3 niveles de la brecha digital?",
    "¿Cómo evitar la deshumanización al usar IA?",
]

CSS = """
<style>
.stApp {background:#F7F8FC;}
html, body, [class*="css"] {font-size:17px;}
h1, h2, h3, h4 {color:#003366;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:4px solid #6F42C1;}
.hero {background:linear-gradient(120deg,#003366 0%,#6F42C1 100%); padding:1.2rem 1.6rem; border-radius:16px; margin-bottom:1rem;}
.hero h1 {color:#fff; margin:0; font-size:1.8rem;}
.hero p {color:#E8E3F7; margin:.3rem 0 0 0; font-size:1.02rem;}
.escudo {width:110px; height:110px; border-radius:50%; background:#003366; color:#fff; border:5px solid #6F42C1;
         display:flex; align-items:center; justify-content:center; text-align:center; font-weight:800; margin:0 auto .6rem auto;}
.pie {text-align:center; color:#4A4A4A; font-size:.85rem; margin-top:1rem;}
.stButton > button {min-height:44px;}
:focus-visible {outline:3px solid #6F42C1 !important; outline-offset:2px;}
@media (prefers-reduced-motion: reduce) {* {animation:none !important; transition:none !important; scroll-behavior:auto !important;}}
</style>
"""


def configurar_pagina() -> None:
    st.set_page_config(page_title="Asistente TIC · UNA Puno", page_icon="🤝", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════ Privacidad (PII) ═══════════════════════════
RE_TEL = re.compile(r"(?<!\d)(?:\+?51[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")  # 9 dígitos
RE_DNI = re.compile(r"(?<!\d)\d{8}(?!\d)")  # 8 dígitos


def enmascarar_pii(texto: str) -> tuple[str, int, int]:
    """Reemplaza teléfonos y DNI. Devuelve (texto, n_dni, n_telefonos)."""
    if not texto:
        return texto, 0, 0
    texto, n_tel = RE_TEL.subn("[TELÉFONO_OCULTO]", texto)
    texto, n_dni = RE_DNI.subn("[DNI_OCULTO]", texto)
    return texto, n_dni, n_tel


def filtrar_pii_stream(flujo):
    """Filtro PII sobre el streaming sin dejar pasar números partidos entre trozos.

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


# ═══════════════════════════ Documento base (PDF) ═══════════════════════════
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


# ═══════════════════════════ Gemini: chat con streaming ═══════════════════════════
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
    doc = contexto[:MAX_CONTEXTO] if contexto else "(No se cargó el documento base: avisa a la persona de que respondes solo con conocimiento general.)"
    privacidad = (
        "El filtro ético de privacidad está ACTIVO: nunca repitas DNI ni teléfonos; usa [DNI_OCULTO] y [TELÉFONO_OCULTO]."
        if pii
        else "Recuerda a la persona que no debe compartir datos reales de casos."
    )
    return f"""Eres el «Asistente de Protocolos TIC» de la Escuela Profesional de Trabajo Social de la UNA Puno.
Responde en español, con tono profesional, claro y accesible (máximo ~250 palabras salvo que pidan más).
Reglas:
1. Basa tus respuestas en el DOCUMENTO BASE y cita a los autores tal como aparecen (Reamer, Castells, van Dijk, Gómez Navarro et al., NASW et al. 2017, FITS 2018, Hwang 2024, etc.).
2. Si algo no está en el documento (p. ej. la Ley N.° 29733), indícalo como conocimiento general y pide verificar la norma vigente.
3. Las TIC complementan, no sustituyen, la empatía, la escucha activa ni el juicio ético profesional.
4. Considera siempre confidencialidad, consentimiento informado y brecha digital (zonas rurales de Puno, lenguas quechua y aimara).
5. {privacidad}
6. Tu respuesta se leerá en voz alta: escribe frases completas y no demasiado largas, y evita tablas.
7. Si la pregunta es ambigua o está incompleta, pide una aclaración breve en lugar de suponer lo que la persona quiso decir.

=== DOCUMENTO BASE ===
{doc}
=== FIN DEL DOCUMENTO ==="""


def _es(texto: str, claves: tuple[str, ...]) -> bool:
    return any(k in texto for k in claves)


_CLAVES_RETIRADO = ("404", "not_found", "no longer available")
_CLAVES_SATURADO = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted")


def _candidatos() -> list[str]:
    return [MODELO] + [m for m in MODELOS_RESPALDO if m != MODELO]


def flujo_gemini(api_key: str, contenido, sistema: str):
    """Generador: entrega el texto de Gemini a medida que se produce (streaming).

    Si el modelo está saturado o retirado ANTES de emitir texto, reintenta y pasa al respaldo.
    Si falla a mitad de la respuesta no se puede cambiar de modelo: se propaga el error.
    """
    cliente = genai.Client(api_key=api_key)
    extra = {"temperature": TEMPERATURA} if TEMPERATURA is not None else {}
    ultimo_error: Exception | None = None
    for modelo in _candidatos():
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


def llamar_con_respaldo(hacer):
    """Para llamadas SIN streaming: hacer(modelo) → resultado, con los mismos reintentos y respaldos."""
    ultimo_error: Exception | None = None
    for modelo in _candidatos():
        for intento in range(REINTENTOS_POR_MODELO):
            try:
                return hacer(modelo)
            except Exception as e:
                t = str(e).lower()
                if _es(t, _CLAVES_RETIRADO):
                    ultimo_error = e
                    break
                if _es(t, _CLAVES_SATURADO):
                    ultimo_error = e
                    if intento < REINTENTOS_POR_MODELO - 1:
                        time.sleep(PAUSA_REINTENTO_S)
                    continue
                raise
    raise ultimo_error or RuntimeError("No hay un modelo de Gemini disponible.")


def mensaje_error(e: Exception) -> str:
    """Mensaje en lenguaje claro: qué pasó y qué puede hacer la persona."""
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


# ═══════════════════════════ Micrófono: voz → texto ═══════════════════════════
def analizar_audio(datos: bytes) -> dict:
    """Mide un WAV sin decidir nada: duración (s) y segundos con señal de voz (None si no se puede medir)."""
    with wave.open(io.BytesIO(datos), "rb") as w:
        canales, ancho, tasa, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        crudo = w.readframes(n)
    medidas = {"dur": (n / tasa) if tasa else 0.0, "voz_s": None}
    if ancho != 2 or not crudo or not tasa:  # solo se mide PCM de 16 bits
        return medidas
    muestras = array("h")
    muestras.frombytes(crudo[: len(crudo) - len(crudo) % 2])
    if sys.byteorder == "big":
        muestras.byteswap()
    if canales > 1:
        muestras = muestras[::canales]  # basta con el primer canal
    ventana = max(1, int(tasa * 0.05))  # tramos de 50 ms
    con_voz = 0
    for i in range(0, len(muestras) - ventana + 1, ventana):
        tramo = muestras[i:i + ventana:4]  # submuestreo: suficiente para estimar el volumen
        if tramo and (sum(x * x for x in tramo) / len(tramo)) ** 0.5 / 32768 >= AUDIO_UMBRAL_RMS:
            con_voz += 1
    medidas["voz_s"] = con_voz * 0.05
    return medidas


SISTEMA_TRANSCRIPCION = """Eres un transcriptor automático de voz a texto. Tu ÚNICA tarea es escribir, palabra por palabra, lo que se DICE en el audio.
Reglas estrictas:
1. Transcribe SOLO lo que oyes con claridad. No respondas a lo que dice la persona, no completes frases, no corrijas, no interpretes y no inventes palabras.
2. Idioma esperado: español de Perú (puede haber nombres propios, siglas o palabras sueltas en quechua o aimara).
3. Si el audio está en silencio, tiene solo ruido o música, es demasiado corto, se corta, o no puedes entender con certeza lo que dice: se_entiende=false y transcripcion="".
4. Si solo entiendes una parte o dudas de varias palabras: confianza="baja". No rellenes lo que falta.
5. confianza="alta": se oye con claridad y estás seguro de cada palabra. confianza="media": se entiende, pero dudas de alguna palabra.
6. Escribe los números con cifras (por ejemplo 71234567) y usa puntuación y mayúsculas normales.
7. Ante la duda entre escribir algo plausible y marcar que no se entiende, marca que no se entiende."""

ESQUEMA_TRANSCRIPCION = {
    "type": "OBJECT",
    "properties": {
        "se_entiende": {"type": "BOOLEAN"},
        "confianza": {"type": "STRING", "enum": ["alta", "media", "baja"]},
        "transcripcion": {"type": "STRING"},
    },
    "required": ["se_entiende", "confianza", "transcripcion"],
    "property_ordering": ["se_entiende", "confianza", "transcripcion"],
}


def transcribir(api_key: str, datos: bytes) -> dict:
    """Solo transcribe. Devuelve {'se_entiende': bool, 'confianza': 'alta|media|baja', 'texto': str}."""
    cliente = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SISTEMA_TRANSCRIPCION,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ESQUEMA_TRANSCRIPCION,
    )
    contenido = [
        types.Content(
            role="user",
            parts=[types.Part.from_bytes(data=datos, mime_type="audio/wav"), types.Part.from_text(text="Transcribe literalmente este audio.")],
        )
    ]
    bruto = llamar_con_respaldo(
        lambda modelo: cliente.models.generate_content(model=modelo, contents=contenido, config=config).text or ""
    )
    bruto = re.sub(r"^```(?:json)?\s*|\s*```$", "", bruto.strip())
    try:
        d = json.loads(bruto)
    except Exception:
        log.warning("Transcripción con formato inesperado: %r", bruto[:200])
        return {"se_entiende": False, "confianza": "baja", "texto": ""}
    confianza = str(d.get("confianza", "baja")).lower()
    return {
        "se_entiende": bool(d.get("se_entiende")),
        "confianza": confianza if confianza in ("alta", "media", "baja") else "baja",
        "texto": str(d.get("transcripcion", "")).strip(),
    }


RELLENOS = {"eh", "ehh", "ah", "ahh", "mm", "mmm", "hm", "hmm", "um", "uh", "oh", "ajá", "aja"}


def evaluar_transcripcion(r: dict, dur: float | None) -> tuple[str, str]:
    """Decide qué hacer con una transcripción: ('ok' | 'dudosa' | 'repetir', motivo)."""
    texto = (r.get("texto") or "").strip()
    palabras = re.findall(r"[a-záéíóúñü0-9]+", texto.lower())
    if not texto or not r.get("se_entiende") or r.get("confianza") == "baja":
        return "repetir", "No se entendió con claridad lo que dijiste."
    if sum(len(p) for p in palabras) < 3 or all(p in RELLENOS for p in palabras):
        return "repetir", "Solo se captaron sonidos sueltos, no una pregunta."
    if dur and len(texto) > dur * 35 + 25:  # imposible hablar tan rápido: el texto no sale del audio
        return "repetir", "La transcripción no cuadra con la duración del audio (posible ruido)."
    if r.get("confianza") != "alta" and CONFIRMAR_SI_DUDA:
        return "dudosa", ""
    return "ok", ""


def procesar_audio(datos: bytes, api_key: str, pii: bool) -> dict | None:
    """Audio → consulta de texto, o None si hay que repetirlo/confirmarlo (deja el aviso en session_state)."""
    ss = st.session_state

    def repetir(motivo: str, oido: str = "") -> None:
        texto = (
            "🎙️ **No logré entender bien tu audio y prefiero no adivinar tu pregunta.** "
            "¿Puedes grabarlo de nuevo? Acércate al micrófono, habla con calma y, si puedes, busca un lugar con menos ruido. "
            "También puedes escribir tu consulta."
        )
        ss["aviso_voz"] = {"texto": texto, "motivo": motivo, "oido": oido}

    try:
        medidas = analizar_audio(datos)
    except Exception as e:  # formato WAV inesperado: no bloquear, la transcripción también valida
        log.warning("No se pudo medir el audio: %r", e)
        medidas = None
    dur = medidas["dur"] if medidas else None
    if medidas:
        if medidas["dur"] < AUDIO_MIN_S:
            repetir(f"El audio dura {medidas['dur']:.1f} s: es demasiado corto.")
            return None
        if medidas["dur"] > AUDIO_MAX_S:
            repetir(f"El audio dura más de {int(AUDIO_MAX_S)} s: haz preguntas más breves.")
            return None
        if medidas["voz_s"] is not None and medidas["voz_s"] < AUDIO_VOZ_MIN_S:
            repetir("Casi no se escucha voz en la grabación (silencio o solo ruido). Revisa que el micrófono esté activo.")
            return None

    with st.spinner("🎤 Transcribiendo tu audio…"):
        try:
            r = transcribir(api_key, datos)
        except Exception as e:
            log.warning("Fallo al transcribir: %r", e)
            ss["aviso_voz"] = {"texto": f"{mensaje_error(e)}\n\nNo pude transcribir tu audio. Inténtalo de nuevo o escribe tu consulta.", "motivo": "", "oido": ""}
            return None

    texto = enmascarar_pii(r["texto"])[0] if pii else r["texto"]
    estado, motivo = evaluar_transcripcion(r, dur)
    if estado == "repetir":
        repetir(motivo, oido=texto)
        return None
    if estado == "dudosa":
        ss["confirmar_voz"] = {"texto": texto}
        return None
    return {"texto": texto, "voz": True}


# ═══════════════════════════ Lectura en voz alta: texto → voz ═══════════════════════════
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
    t = re.sub(r"<[^<>\n]{1,25}>", " ", t)  # <br>…: no deben leerse
    t = t.replace("<", " ").replace(">", " ")
    for patron, lectura in PRONUNCIACION.items():
        t = re.sub(patron, lectura, t)
    t = re.sub(r"(?<!\.)\.\.(?!\.)", ".", t)  # «N.A.S.W.» + punto final no debe quedar con doble punto
    lineas = []
    for linea in t.splitlines():
        linea = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", linea)).strip()
        if linea:  # una pausa natural al final de cada línea
            lineas.append(linea if linea[-1] in ".,;:!?…)" else linea + ".")
    t = "\n".join(lineas)
    if len(t) > MAX_CHARS_VOZ:  # se corta en una frase completa
        corte = max(t.rfind(". ", 0, MAX_CHARS_VOZ), t.rfind("\n", 0, MAX_CHARS_VOZ))
        t = t[: corte + 1] if corte > MAX_CHARS_VOZ // 2 else t[:MAX_CHARS_VOZ]
    return t


async def _edge_async(texto: str, voz: str, rate: str) -> bytes:
    audio = bytearray()
    async for trozo in edge_tts.Communicate(texto, voz, rate=rate).stream():
        if trozo["type"] == "audio":
            audio.extend(trozo["data"])
    if not audio:
        raise RuntimeError("edge-tts no devolvió audio")
    return bytes(audio)


def _correr(coro, timeout: float):
    """Ejecuta una corrutina en un hilo propio (Streamlit no tiene bucle asyncio en su hilo de script)."""
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(asyncio.run, coro).result(timeout=timeout)
    finally:
        ex.shutdown(wait=False)


def _gtts(texto: str, lenta: bool) -> bytes:
    buf = io.BytesIO()
    gTTS(text=texto, lang="es", tld=GTTS_TLD, slow=lenta).write_to_fp(buf)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=40, ttl=3600)
def sintetizar(texto: str, voz: str, velocidad: float) -> tuple[bytes, str]:
    """Texto → MP3. Devuelve (audio, motor). Prueba edge-tts (2 intentos) y luego gTTS. Los fallos no se guardan en caché."""
    rate = f"{round((velocidad - 1) * 100):+d}%"
    errores: list[str] = []
    if edge_tts is not None:
        for _ in range(2):
            try:
                return _correr(_edge_async(texto, voz, rate), TTS_TIMEOUT_S), "Microsoft Edge (voz neuronal)"
            except Exception as e:
                errores.append(f"edge-tts: {type(e).__name__}: {e}")
                time.sleep(0.5)
    if gTTS is not None:
        try:
            return _gtts(texto, velocidad < 0.95), "Google (voz alternativa)"
        except Exception as e:
            errores.append(f"gTTS: {type(e).__name__}: {e}")
    raise RuntimeError(" | ".join(errores) or "No hay ninguna librería de voz instalada (pip install edge-tts gTTS).")


def mostrar_audio(texto: str, cfg: dict, autoplay: bool) -> None:
    """Genera el MP3 y lo muestra en un reproductor estándar (funciona en cualquier navegador)."""
    hablado = texto_para_voz(texto)
    if not hablado:
        return
    try:
        with st.spinner("🔊 Preparando la voz…"):
            audio, motor = sintetizar(hablado, cfg["voz"], cfg["velocidad"])
    except Exception as e:
        log.warning("No se pudo generar la voz: %s", e)
        st.caption("🔇 No pude preparar la voz ahora mismo (¿sin conexión del servidor?). Puedes volver a intentarlo con «Escuchar».")
        return
    try:
        st.audio(audio, format="audio/mpeg", autoplay=autoplay)
    except TypeError:  # Streamlit antiguo sin «autoplay»
        st.audio(audio, format="audio/mpeg")
    st.caption(f"🔊 {motor}" + (" · si no suena sola, pulsa ▶ (algunos navegadores bloquean el audio automático)" if autoplay else ""))


# ═══════════════════════════ Interfaz ═══════════════════════════
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
        st.divider()
        st.markdown("**🔊 Lectura en voz alta**")
        st.toggle(
            "Leer cada respuesta al terminar",
            value=VOZ_INICIAL and VOZ_OK,
            key="voz_on",
            disabled=not VOZ_OK,
            help="Al terminar de escribirse, la respuesta se reproduce sola. «Escuchar» sigue disponible aunque esto esté apagado.",
        )
        st.selectbox("Voz", list(VOCES), index=list(VOCES).index(VOZ_POR_DEFECTO), key="voz_sel", disabled=not VOZ_OK)
        st.select_slider(
            "Velocidad",
            options=list(VOZ_VELOCIDADES),
            value=VOZ_VELOCIDAD_INICIAL,
            key="voz_vel",
            disabled=not VOZ_OK,
            help="Se aplica a los audios que se generen a partir de ahora.",
        )
        if VOZ_OK:
            st.caption("Para generar el audio, el texto de la respuesta se envía a Microsoft (o a Google, si se usa la voz alternativa). Con el filtro de privacidad activo no incluye DNI ni teléfonos.")
        else:
            st.warning("Falta instalar una librería de voz: pip install edge-tts gTTS")
        st.divider()
        contexto, estado, ok = obtener_contexto()
        (st.success if ok else st.warning)(estado)
        st.markdown(
            "<div class='pie'>© 2026 · Escuela Profesional de Trabajo Social<br>Universidad Nacional del Altiplano - Puno</div>",
            unsafe_allow_html=True,
        )
    return pii, contexto


def config_voz() -> dict:
    ss = st.session_state
    return dict(
        activa=bool(ss.get("voz_on", VOZ_INICIAL)) and VOZ_OK,
        voz=VOCES.get(ss.get("voz_sel", VOZ_POR_DEFECTO), VOCES[VOZ_POR_DEFECTO]),
        velocidad=VOZ_VELOCIDADES.get(ss.get("voz_vel", VOZ_VELOCIDAD_INICIAL), 1.0),
    )


# ── Estado y callbacks ──
def _saludo() -> dict:
    return {
        "role": "assistant",
        "content": "¡Hola! Soy el asistente de protocolos TIC. Pregúntame sobre herramientas digitales, ética, brecha digital o cómo humanizar la intervención. Puedes escribir o grabar tu pregunta 🎙️.",
    }


def _nueva_conversacion() -> None:
    ss = st.session_state
    ss.messages = [_saludo()]
    for k in ("fallida", "aviso_voz", "confirmar_voz", "voz_confirmada", "escuchar", "reintentar", "pregunta_rapida"):
        ss.pop(k, None)
    ss["audio_n"] = ss.get("audio_n", 0) + 1  # vacía el grabador


def _pedir_escuchar(i: int) -> None:
    st.session_state["escuchar"] = i


def _marcar_reintento() -> None:
    st.session_state["reintentar"] = True


def _aceptar_voz() -> None:
    ss = st.session_state
    ss["voz_confirmada"] = ss.pop("confirmar_voz")["texto"]


def _rechazar_voz() -> None:
    st.session_state.pop("confirmar_voz", None)


def boton_escuchar(i: int) -> None:
    st.button("🔊 Escuchar", key=f"escuchar_{i}", on_click=_pedir_escuchar, args=(i,), help="Lee esta respuesta en voz alta.")


# ── Pintado ──
def pintar_historial(pii: bool, cfg: dict) -> None:
    ss = st.session_state
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(f"🎤 *Pregunta por voz:* «{m['content']}»" if m.get("voz") else m["content"])
            if m.get("nota"):
                st.caption(m["nota"])
            if m["role"] == "assistant":
                boton_escuchar(i)
                if ss.get("escuchar") == i:  # el audio se muestra una sola vez (no se repite en cada actualización)
                    mostrar_audio(enmascarar_pii(m["content"])[0] if pii else m["content"], cfg, autoplay=True)
    ss.pop("escuchar", None)


def pintar_fallo(f: dict) -> None:
    """El error dice qué pasó y qué hacer, no se pierde la pregunta y se puede reintentar con un clic."""
    with st.chat_message("user"):
        st.markdown(f["resumen"])
    with st.chat_message("assistant"):
        st.error(f["mensaje"])
        with st.expander("🔧 Detalles técnicos (para quien administra la aplicación)"):
            st.code(f["detalle"], language=None)
        st.button("🔄 Reintentar", key="btn_reintentar", on_click=_marcar_reintento, type="primary")


def pintar_aviso_voz(a: dict) -> None:
    with st.chat_message("assistant"):
        st.warning(a["texto"])
        if a.get("oido"):
            st.caption(f"Lo que alcancé a captar: «{a['oido']}» (no es suficiente para responder con seguridad).")
        if a.get("motivo"):
            st.caption(f"Motivo: {a['motivo']}")


def pintar_confirmacion(c: dict) -> None:
    with st.chat_message("assistant"):
        st.info(f"🎤 Creo que dijiste: **«{c['texto']}»**\n\nNo estoy del todo seguro de haberte entendido bien. ¿Es correcto?")
        si, no = st.columns(2)
        si.button("✅ Sí, responder", key="voz_si", on_click=_aceptar_voz, type="primary")
        no.button("🎙️ No, grabaré de nuevo", key="voz_no", on_click=_rechazar_voz)


# ── Un turno completo ──
def _emitir(flujo, acum: dict, ph_estado):
    primero = True
    for trozo in flujo:
        if primero:
            ph_estado.empty()  # el aviso «Consultando…» se va al llegar el primer texto
            primero = False
        acum["texto"] += trozo
        yield trozo


def responder(consulta: dict, contexto: str, api_key: str, pii: bool, cfg: dict) -> None:
    """Burbuja de la persona → respuesta en streaming → historial → voz."""
    ss = st.session_state
    voz = bool(consulta.get("voz"))
    pregunta, n_dni, n_tel = enmascarar_pii(consulta["texto"]) if pii else (consulta["texto"], 0, 0)
    nota = f"🔒 Filtro de privacidad: {n_dni} DNI y {n_tel} teléfono(s) ocultados antes de enviar." if (n_dni or n_tel) else None
    resumen = f"🎤 *Pregunta por voz:* «{pregunta}»" if voz else pregunta

    sistema = construir_sistema(contexto, pii)
    previos = ss.messages[-HISTORIAL_MENSAJES:]
    while previos and previos[0]["role"] != "user":
        previos = previos[1:]
    contenido = [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])])
        for m in previos
    ] + [types.Content(role="user", parts=[types.Part.from_text(text=pregunta)])]

    with st.chat_message("user"):  # la pregunta aparece al instante
        st.markdown(resumen)
        if nota:
            st.caption(nota)

    ss.pop("modelo_usado", None)
    acum = {"texto": ""}
    with st.chat_message("assistant"):
        ph_estado = st.empty()
        ph_estado.caption("⏳ Consultando el documento base…")
        flujo = flujo_gemini(api_key, contenido, sistema)
        if pii:
            flujo = filtrar_pii_stream(flujo)
        error: Exception | None = None
        try:
            st.write_stream(_emitir(flujo, acum, ph_estado))
        except Exception as e:
            error = e
        ph_estado.empty()

        respuesta = acum["texto"].strip()
        if not respuesta and error is None:
            error = RuntimeError("La respuesta llegó vacía.")
        if error is not None:
            log.warning("Fallo en la consulta: %r", error)
            if not respuesta:  # no se mostró nada: se guarda la consulta para «Reintentar»
                ss["fallida"] = dict(
                    consulta=dict(texto=pregunta, voz=voz), resumen=resumen,
                    mensaje=mensaje_error(error), detalle=f"{type(error).__name__}: {error}",
                )
                st.rerun()
        aviso = f"⚠️ La respuesta se interrumpió antes de terminar. {mensaje_error(error)}" if error is not None else None

        usado = ss.get("modelo_usado", MODELO)
        nota_ia = aviso or (f"ℹ️ Respondió {usado}: el modelo principal estaba saturado o no disponible." if usado != MODELO else None)
        if nota_ia:
            st.caption(nota_ia)

        ss.messages.append({"role": "user", "content": pregunta, "voz": voz, "nota": nota})
        ss.messages.append({"role": "assistant", "content": respuesta, "nota": nota_ia})
        ss.pop("fallida", None)
        boton_escuchar(len(ss.messages) - 1)
        if cfg["activa"] and error is None:
            mostrar_audio(respuesta, cfg, autoplay=True)


# ── Pantalla del chat ──
def pantalla_chat(contexto: str, api_key: str, pii: bool, cfg: dict) -> None:
    ss = st.session_state
    listo = GENAI_OK and bool(api_key)

    cab, parar, borrar = st.columns([4, 1.4, 1.8], vertical_alignment="center")
    cab.subheader("🤖 Asistente de Protocolos TIC")
    parar.button("⏹ Detener voz", key="btn_parar", help="Corta el audio que esté sonando.")  # al recargar, el reproductor desaparece
    with borrar.popover("🗑️ Nueva conversación"):
        st.write("Se borrará el historial de esta sesión.")
        st.button("Sí, borrar", key="btn_borrar", type="primary", on_click=_nueva_conversacion)
    st.caption("Responde con base en la monografía del curso (Reamer, Castells, van Dijk, NASW/FITS). Puedes escribir o hablar, y escuchar la respuesta.")
    with st.expander("ℹ️ ¿Cómo usar este asistente?"):
        st.markdown(
            "1. **Escribe, graba con el micrófono o elige** una pregunta rápida.\n"
            "2. La respuesta **aparece mientras se genera** y, al terminar, **se lee en voz alta**. Si tu navegador bloquea el audio automático, pulsa **▶** en el reproductor.\n"
            "3. **⏹ Detener voz** corta el audio; **🔊 Escuchar** repite cualquier respuesta. Voz y velocidad se cambian en el menú lateral.\n"
            "4. Si tu audio **no se entiende bien**, te pediré que lo repitas (o que confirmes lo que entendí) en lugar de adivinar.\n"
            "5. **No escribas ni dictes datos reales de casos.** El filtro oculta DNI y teléfonos, pero no nombres ni direcciones.\n\n"
            "*El micrófono necesita permiso del navegador y funciona en páginas seguras (https) o en localhost.*"
        )

    if not GENAI_OK:
        st.error("🔌 El asistente no está disponible: falta instalar un componente en el servidor (pip install google-genai). Avisa a quien administra la aplicación.")
    elif not api_key:
        st.error("🔑 El asistente aún no está activado: falta la clave del servicio de IA (GEMINI_API_KEY en .streamlit/secrets.toml o como variable de entorno).")
    if not contexto:
        st.warning("📄 Sin documento base: el asistente responderá solo con conocimiento general.")

    zona = st.container()  # historial + respuesta en curso: queda ARRIBA de los controles de entrada

    st.markdown("**💡 Preguntas rápidas**")
    for col, pregunta in zip(st.columns(len(EJEMPLOS)), EJEMPLOS):
        if col.button(pregunta, key=f"ej_{pregunta}", disabled=not listo):
            ss.pregunta_rapida = pregunta

    audio = None
    if listo and hasattr(st, "audio_input"):
        st.markdown("**🎙️ Pregunta con tu voz**")
        audio = st.audio_input("Graba tu pregunta", key=f"audio_{ss.audio_n}", label_visibility="collapsed")
        st.caption(
            "Habla con calma y cerca del micrófono. Si no te entiendo bien, te pediré repetirlo. "
            + ("🔒 El audio no puede filtrarse antes de enviarse: no dictes datos reales (la transcripción y la respuesta sí se filtran)." if pii else "")
        )
    texto = st.chat_input("Escribe tu consulta sobre TIC en Trabajo Social…", max_chars=MAX_CONSULTA, disabled=not listo)
    texto = texto or ss.pop("pregunta_rapida", None)

    # ¿Qué hay que atender en esta ejecución? (los avisos de un turno anterior se descartan al llegar algo nuevo)
    reintentar = ss.pop("reintentar", False)
    confirmada = ss.pop("voz_confirmada", None)
    consulta: dict | None = None
    datos_audio: bytes | None = None
    if texto:
        consulta = {"texto": texto, "voz": False}
    elif audio is not None:
        datos = audio.getvalue()
        huella = hashlib.sha256(datos).hexdigest()
        if huella != ss.get("audio_visto"):  # la misma grabación no se procesa dos veces
            ss["audio_visto"] = huella
            ss["audio_n"] += 1  # el grabador se vacía en la próxima actualización, listo para otra pregunta
            datos_audio = datos
    if consulta is None and datos_audio is None:
        if reintentar and ss.get("fallida"):
            consulta = dict(ss["fallida"]["consulta"])
        elif confirmada:
            consulta = {"texto": confirmada, "voz": True}
    if consulta is not None or datos_audio is not None:
        for k in ("fallida", "aviso_voz", "confirmar_voz"):
            ss.pop(k, None)

    with zona:
        pintar_historial(pii, cfg)
        if datos_audio is not None and listo:
            consulta = procesar_audio(datos_audio, api_key, pii)  # None = hay que repetir o confirmar
        if consulta is not None and listo:
            responder(consulta, contexto, api_key, pii, cfg)
        elif ss.get("confirmar_voz"):
            pintar_confirmacion(ss["confirmar_voz"])
        elif ss.get("aviso_voz"):
            pintar_aviso_voz(ss["aviso_voz"])
        elif ss.get("fallida"):
            pintar_fallo(ss["fallida"])


# ═══════════════════════════ Aplicación ═══════════════════════════
def main() -> None:
    configurar_pagina()
    ss = st.session_state
    ss.setdefault("messages", [_saludo()])
    ss.setdefault("audio_n", 0)
    pii, contexto = barra_lateral()
    api_key = clave_api() if GENAI_OK else ""
    cfg = config_voz()

    st.markdown(
        '<div class="hero"><h1>Asistente de Protocolos TIC</h1>'
        "<p>Tecnología ética, accesible e inclusiva para la intervención en Trabajo Social: "
        "pregunta escribiendo o hablando, y escucha la respuesta.</p></div>",
        unsafe_allow_html=True,
    )
    pantalla_chat(contexto, api_key, pii, cfg)


if __name__ == "__main__":
    main()
