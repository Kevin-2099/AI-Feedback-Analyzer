import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

# ============================================================================
# 1) LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("feedback_analyzer")


# ============================================================================
# 2) INTERRUPTORES DE FUNCIONALIDAD
# ============================================================================
# Cargar 4-5 modelos de IA a la vez puede consumir varios GB de RAM. Estos
# interruptores permiten desactivar las funciones mas pesadas sin tocar
# el codigo, por ejemplo en un servidor con recursos limitados.
ENABLE_EMOTION = os.environ.get("FA_ENABLE_EMOTION", "1") != "0"
ENABLE_TOXICITY = os.environ.get("FA_ENABLE_TOXICITY", "1") != "0"
ENABLE_DEDUP = os.environ.get("FA_ENABLE_DEDUP", "1") != "0"

DB_PATH = os.environ.get("FA_DB_PATH", "feedback_history.db")


# ============================================================================
# 3) BASE DE DATOS - HISTORIAL PERSISTENTE 
# ============================================================================
def init_db() -> None:
    """Crea la tabla de historial si no existe todavia."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT,
                    sentiment TEXT,
                    emotion TEXT,
                    issues TEXT,
                    urgency TEXT,
                    toxic INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()
    except Exception as e:
        logger.error("No se pudo inicializar la base de datos: %s", e)


def save_to_history(
    source: str,
    text: str,
    language: str,
    sentiment: str,
    emotion: str,
    issues: str,
    urgency: str,
    toxic: bool = False,
) -> None:
    """Guarda un analisis individual en el historial SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO history
                   (timestamp, source, text, language, sentiment, emotion, issues, urgency, toxic)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    source,
                    text[:500],
                    language,
                    sentiment,
                    emotion,
                    issues,
                    urgency,
                    int(toxic),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.error("No se pudo guardar en el historial: %s", e)


def load_history(limit: int = 200) -> pd.DataFrame:
    """Devuelve los ultimos `limit` analisis guardados, mas recientes primero."""
    columns = ["timestamp", "source", "text", "language", "sentiment", "emotion", "issues", "urgency", "toxic"]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f"SELECT {', '.join(columns)} FROM history ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return df
    except Exception as e:
        logger.error("No se pudo leer el historial: %s", e)
        return pd.DataFrame(columns=columns)


def history_trend() -> pd.DataFrame:
    """Agrega el historial por fecha y sentimiento, para visualizar tendencias."""
    columns = ["fecha", "sentimiento_simple", "conteo"]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT timestamp, sentiment FROM history", conn)
        if df.empty:
            return pd.DataFrame(columns=columns)
        df["fecha"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        df["sentimiento_simple"] = df["sentiment"].str.extract(r"(Positivo|Neutral|Negativo)")
        df["sentimiento_simple"] = df["sentimiento_simple"].fillna("Neutral")
        trend = df.groupby(["fecha", "sentimiento_simple"]).size().reset_index(name="conteo")
        return trend
    except Exception as e:
        logger.error("No se pudo calcular la tendencia del historial: %s", e)
        return pd.DataFrame(columns=columns)


# ============================================================================
# 4) CARGA PEREZOSA DE MODELOS 
# ============================================================================
# Ningun modelo se carga hasta que realmente se necesita, y cada carga se
# cachea para no repetirla. Si una libreria no esta instalada o un modelo
# no se puede descargar, la funcion correspondiente devuelve None y el
# resto de la app sigue funcionando con un mensaje explicativo en vez de
# romperse.
_MODELS: Dict[str, object] = {}


def get_sentiment_pipeline():
    if "sentiment" not in _MODELS:
        try:
            from transformers import pipeline

            logger.info("Cargando modelo de sentimiento...")
            _MODELS["sentiment"] = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
            )
        except Exception as e:
            logger.error("No se pudo cargar el modelo de sentimiento: %s", e)
            _MODELS["sentiment"] = None
    return _MODELS["sentiment"]


def get_summarizer():
    if "summarizer" not in _MODELS:
        try:
            from transformers import pipeline

            logger.info("Cargando modelo de resumen...")
            _MODELS["summarizer"] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
            )
        except Exception as e:
            logger.error("No se pudo cargar el modelo de resumen: %s", e)
            _MODELS["summarizer"] = None
    return _MODELS["summarizer"]


def get_emotion_analyzer(lang: str):
    """Analizador de emociones de pysentimiento (robertuito para 'es', equivalente para 'en')."""
    key = f"emotion_{lang}"
    if key not in _MODELS:
        try:
            from pysentimiento import create_analyzer

            logger.info("Cargando analizador de emociones (%s)...", lang)
            _MODELS[key] = create_analyzer(task="emotion", lang=lang)
        except Exception as e:
            logger.warning("Analizador de emociones (%s) no disponible: %s", lang, e)
            _MODELS[key] = None
    return _MODELS[key]


def get_toxicity_model():
    if "toxicity" not in _MODELS:
        try:
            from detoxify import Detoxify

            logger.info("Cargando modelo de toxicidad (multilingual)...")
            _MODELS["toxicity"] = Detoxify("multilingual")
        except Exception as e:
            logger.warning("Modelo de toxicidad no disponible: %s", e)
            _MODELS["toxicity"] = None
    return _MODELS["toxicity"]


def get_embedding_model():
    if "embeddings" not in _MODELS:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Cargando modelo de embeddings para deduplicacion...")
            _MODELS["embeddings"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            logger.warning("Modelo de embeddings no disponible: %s", e)
            _MODELS["embeddings"] = None
    return _MODELS["embeddings"]


# ============================================================================
# 5) IDIOMA Y SENTIMIENTO
# ============================================================================
def detect_language(text: str) -> Tuple[str, Optional[str]]:
    """Devuelve (texto para mostrar, codigo de idioma) o (mensaje, None) si falla."""
    try:
        from langdetect import detect

        lang = detect(text)
    except Exception as e:
        logger.warning("No se pudo detectar el idioma: %s", e)
        return "Idioma no detectado", None

    if lang == "es":
        return "\U0001f1ea\U0001f1f8 Espanol", "es"
    elif lang == "en":
        return "\U0001f1fa\U0001f1f8 Ingles", "en"
    return f"\U0001f30d {lang}", lang


def map_sentiment(label: str) -> str:
    """Convierte la etiqueta '1 star'..'5 stars' del modelo en una categoria legible."""
    try:
        stars = int(label[0])
    except Exception:
        return "\u2753 Desconocido"
    if stars <= 2:
        return "\U0001f621 Negativo"
    elif stars == 3:
        return "\U0001f610 Neutral"
    return "\U0001f600 Positivo"


SENTIMENT_KEYS = ("Positivo", "Neutral", "Negativo")


def _extract_sentiment_key(sentiment_text: str) -> str:
    """Extrae 'Positivo'/'Neutral'/'Negativo' de un texto tipo 'Positivo (confianza 0.9)'."""
    for key in SENTIMENT_KEYS:
        if key in sentiment_text:
            return key
    return "Neutral"


# ============================================================================
# 6) EMOCIONES 
# ============================================================================
EMOTION_LABELS = {
    "joy": "\U0001f604 Alegria",
    "anger": "\U0001f620 Enojo",
    "sadness": "\U0001f622 Tristeza",
    "fear": "\U0001f628 Miedo",
    "disgust": "\U0001f922 Asco",
    "surprise": "\U0001f632 Sorpresa",
    "others": "\U0001f610 Neutral/Otros",
    "neutral": "\U0001f610 Neutral/Otros",
}


def analyze_emotion(text: str, lang_code: Optional[str]) -> Optional[str]:
    """
    Emociones granulares con pysentimiento, entrenado directamente en el
    idioma del texto (sin traducir antes). Soporta 'es' y 'en'; para otros
    idiomas devuelve None y el llamador debe usar el sentimiento generico.
    """
    if not ENABLE_EMOTION or lang_code not in ("es", "en"):
        return None
    analyzer = get_emotion_analyzer(lang_code)
    if analyzer is None:
        return None
    try:
        result = analyzer.predict(text[:512])
        label = EMOTION_LABELS.get(result.output, f"\U0001f300 {result.output}")
        if result.probas:
            confidence = round(max(result.probas.values()), 2)
            return f"{label} (confianza {confidence})"
        return label
    except Exception as e:
        logger.error("Error en analisis de emociones: %s", e)
        return None


# ============================================================================
# 7) PROBLEMAS DETECTADOS Y RECOMENDACIONES (logica original, sin cambios)
# ============================================================================
NO_ISSUES = "No se detectaron problemas claros"

issues_keywords = {
    "UI confusa": [
        "confusing interface", "hard to navigate", "interface not intuitive", "poor layout",
        "navigation is confusing", "menu is confusing", "difficult navigation",
        "interfaz confusa", "interfaz poco clara", "interfaz no intuitiva",
        "navegacion confusa", "diseno confuso", "menu confuso", "interfaz complicada",
    ],
    "Problemas de login": [
        "cannot login", "can't login", "unable to login", "login not working", "login failed",
        "authentication failed", "invalid credentials", "password not accepted", "login loop",
        "no puedo iniciar sesion", "error al iniciar sesion", "credenciales invalidas",
        "fallo de autenticacion", "problema al iniciar sesion", "bucle de login",
    ],
    "Errores o bugs": [
        "app crashes", "application crashes", "keeps crashing", "random crash",
        "crash when opening", "unexpected error", "internal server error", "error message appears",
        "bug in the app", "major bug", "critical bug", "function not working", "button not working",
        "la app se cierra", "error inesperado", "mensaje de error", "bug en la aplicacion",
        "la app se congela", "funcion no funciona", "boton no funciona", "falla critica",
    ],
    "Problemas de rendimiento": [
        "app is slow", "very slow app", "slow loading", "slow performance", "laggy interface",
        "takes forever to load", "long loading time", "performance issues", "slow response time",
        "aplicacion lenta", "muy lenta", "tiempo de carga largo", "tarda mucho en cargar",
        "rendimiento lento", "interfaz lenta", "retraso en respuesta", "experiencia lenta",
    ],
    "Problemas de soporte": [
        "support not responding", "no response from support", "support never replied",
        "customer support not helpful", "bad customer support", "support ignored my message",
        "waiting for support response", "slow response from support",
        "soporte no responde", "soporte muy lento", "no recibi respuesta del soporte",
        "mala atencion al cliente", "soporte ineficiente", "tiempos de respuesta largos",
    ],
    "Problemas de actualizacion": [
        "update broke the app", "update caused problems", "after update it crashes",
        "latest update broken", "update ruined the app", "update introduced bugs",
        "la actualizacion rompio la app", "problema despues de actualizar",
        "la ultima actualizacion falla", "actualizacion genero errores", "actualizacion inestable",
    ],
}


def extract_issues(text: str) -> List[str]:
    found = []
    lower_text = text.lower()
    for issue, keywords in issues_keywords.items():
        for k in keywords:
            words = k.lower().split()
            if all(word in lower_text for word in words):
                found.append(issue)
                break
    return found if found else [NO_ISSUES]


def generate_recommendations(issues: List[str]) -> List[str]:
    mapping = {
        "UI confusa": "Mejorar navegacion y claridad de la interfaz",
        "Problemas de login": "Revisar sistema de autenticacion y flujo de login",
        "Errores o bugs": "Priorizar correccion de errores criticos",
        "Problemas de rendimiento": "Optimizar rendimiento y tiempos de carga",
        "Problemas de soporte": "Mejorar tiempos de respuesta del soporte",
        "Problemas de actualizacion": "Revisar estabilidad de las ultimas actualizaciones",
    }
    recs = [mapping[issue] for issue in issues if issue in mapping]
    return recs if recs else ["Seguir recopilando feedback de usuarios"]


# ============================================================================
# 8) TOXICIDAD
# ============================================================================
TOXICITY_THRESHOLD = 0.5


def analyze_toxicity(text: str) -> Tuple[bool, str]:
    """Deteccion de toxicidad/lenguaje ofensivo multilenguaje con Detoxify."""
    if not ENABLE_TOXICITY:
        return False, "Desactivado (FA_ENABLE_TOXICITY=0)"
    model = get_toxicity_model()
    if model is None:
        return False, "No disponible (instala `detoxify`)"
    try:
        scores = model.predict(text[:512])
        scores = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in scores.items()}
        flagged = {k: round(v, 2) for k, v in scores.items() if v >= TOXICITY_THRESHOLD}
        if flagged:
            detail = ", ".join(f"{k} ({v})" for k, v in flagged.items())
            return True, f"\u26a0\ufe0f Posible lenguaje ofensivo: {detail}"
        return False, "\u2705 Sin lenguaje ofensivo detectado"
    except Exception as e:
        logger.error("Error en analisis de toxicidad: %s", e)
        return False, "No se pudo evaluar la toxicidad"


# ============================================================================
# 9) SCORE DE URGENCIA 
# ============================================================================
ISSUE_SEVERITY = {
    "Problemas de login": 3,
    "Errores o bugs": 3,
    "Problemas de actualizacion": 2,
    "Problemas de rendimiento": 2,
    "Problemas de soporte": 2,
    "UI confusa": 1,
}

SENTIMENT_WEIGHT = {"Negativo": 3, "Neutral": 1, "Positivo": 0}


def calculate_urgency(
    sentiment_text: str,
    issues: List[str],
    frequency: Optional[int] = None,
    is_toxic: bool = False,
) -> str:
    """
    Combina sentimiento + severidad del tipo de problema (+ frecuencia en
    modo lote) en una etiqueta de urgencia para priorizar que feedback
    atender primero.
    """
    score = SENTIMENT_WEIGHT.get(_extract_sentiment_key(sentiment_text), 1)
    real_issues = [i for i in issues if i != NO_ISSUES]
    if real_issues:
        score += max(ISSUE_SEVERITY.get(i, 1) for i in real_issues)
    if frequency is not None and frequency >= 3:
        score += 1
    if is_toxic:
        score += 1

    if score >= 5:
        return "\U0001f534 Alta"
    elif score >= 3:
        return "\U0001f7e1 Media"
    return "\U0001f7e2 Baja"


# ============================================================================
# 10) DEDUPLICACION CON EMBEDDINGS 
# ============================================================================
DUPLICATE_SIMILARITY_THRESHOLD = 0.92


def deduplicate_texts(texts: List[str]) -> Tuple[List[int], Dict[int, int]]:
    """
    Detecta feedback casi-identico usando embeddings multilingues.
    Devuelve (indices a conservar, {indice_duplicado: indice_original}).
    Si el modelo no esta disponible, no deduplica nada (conserva todo).
    """
    if not ENABLE_DEDUP or len(texts) < 2:
        return list(range(len(texts))), {}

    model = get_embedding_model()
    if model is None:
        return list(range(len(texts))), {}

    try:
        import numpy as np

        embeddings = model.encode(texts, normalize_embeddings=True)
        keep_indices: List[int] = []
        duplicate_map: Dict[int, int] = {}
        for i, emb in enumerate(embeddings):
            is_dup = False
            for j in keep_indices:
                similarity = float(np.dot(emb, embeddings[j]))  # normalizados => coseno
                if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                    duplicate_map[i] = j
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
        return keep_indices, duplicate_map
    except Exception as e:
        logger.error("Error en deduplicacion por embeddings: %s", e)
        return list(range(len(texts))), {}


# ============================================================================
# 11) ANALISIS INDIVIDUAL
# ============================================================================
def analyze_feedback(text: str, record_history: bool = True):
    empty_issues = pd.DataFrame({"Problema detectado": []})
    empty_recs = pd.DataFrame({"Recomendacion": []})
    if not text or not text.strip():
        return "Sin texto", "", "", "", empty_issues, empty_recs, "", ""

    language_display, lang_code = detect_language(text)

    sentiment_pipeline = get_sentiment_pipeline()
    sentiment_result = "Modelo de sentimiento no disponible (instala `transformers`)"
    if sentiment_pipeline is not None:
        try:
            sentiment = sentiment_pipeline(text[:512])[0]
            label = map_sentiment(sentiment["label"])
            confidence = round(sentiment["score"], 2)
            sentiment_result = f"{label} (confianza {confidence})"
        except Exception as e:
            logger.error("Error en analisis de sentimiento: %s", e)
            sentiment_result = "Error al analizar sentimiento"

    emotion_result = analyze_emotion(text, lang_code)
    if emotion_result is None:
        emotion_result = (
            "Desactivado (FA_ENABLE_EMOTION=0)"
            if not ENABLE_EMOTION
            else "No disponible para este idioma / modelo no instalado"
        )

    summarizer = get_summarizer()
    summary = "Modelo de resumen no disponible (instala `transformers`)"
    if summarizer is not None:
        try:
            summary = summarizer(text[:2000], max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
        except Exception as e:
            logger.error("Error al generar resumen: %s", e)
            summary = "No se pudo generar resumen"

    issues = extract_issues(text)
    real_issues = [i for i in issues if i != NO_ISSUES]
    issues_df = pd.DataFrame({"Problema detectado": issues})

    recs = generate_recommendations(issues)
    recs_df = pd.DataFrame({"Recomendacion": recs})

    is_toxic, toxicity_detail = analyze_toxicity(text)
    urgency = calculate_urgency(sentiment_result, real_issues, is_toxic=is_toxic)

    if record_history:
        save_to_history(
            source="individual",
            text=text,
            language=language_display,
            sentiment=sentiment_result,
            emotion=emotion_result,
            issues=", ".join(real_issues),
            urgency=urgency,
            toxic=is_toxic,
        )

    return language_display, sentiment_result, emotion_result, summary, issues_df, recs_df, urgency, toxicity_detail


# ============================================================================
# 12) ANALISIS POR LOTES (CSV/Excel) 
# ============================================================================
TEXT_COLUMN_CANDIDATES = [
    "feedback", "review", "reviews", "text", "texto", "comentario", "comentarios",
    "comment", "comments", "opinion", "opinion", "resena", "resenas",
]


def _find_text_column(df: pd.DataFrame) -> Optional[str]:
    lower_cols = {str(c).lower(): c for c in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def process_batch(file, progress: "gr.Progress" = gr.Progress()):
    if file is None:
        return "\u26a0\ufe0f Sube un archivo CSV o Excel primero.", pd.DataFrame(), None

    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(str(path))[1].lower()

    try:
        if ext == ".csv":
            df_in = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df_in = pd.read_excel(path)
        else:
            return f"\u274c Formato no soportado: '{ext}'. Usa .csv o .xlsx.", pd.DataFrame(), None
    except Exception as e:
        logger.error("Error al leer el archivo de lotes: %s", e)
        return f"\u274c No se pudo leer el archivo: {e}", pd.DataFrame(), None

    text_column = _find_text_column(df_in)
    if text_column is None:
        return (
            "\u274c No se encontro una columna de texto reconocible "
            "(p. ej. 'feedback', 'review', 'comentario').",
            pd.DataFrame(),
            None,
        )

    raw_texts = [t for t in df_in[text_column].astype(str).fillna("").tolist() if t.strip()]
    if not raw_texts:
        return "\u274c El archivo no contiene texto para analizar.", pd.DataFrame(), None

    keep_indices, duplicate_map = deduplicate_texts(raw_texts)
    n_duplicates = len(duplicate_map)

    analyzed = []
    issue_counter: Dict[str, int] = {}
    lang_counter: Dict[str, int] = {}
    sentiment_counter = {"Positivo": 0, "Neutral": 0, "Negativo": 0}
    toxic_count = 0

    total = len(keep_indices)
    sentiment_pipeline = get_sentiment_pipeline()

    for progress_i, idx in enumerate(keep_indices):
        text = raw_texts[idx]
        progress((progress_i + 1) / total, desc=f"Analizando {progress_i + 1}/{total}")

        language_display, lang_code = detect_language(text)
        lang_counter[language_display] = lang_counter.get(language_display, 0) + 1

        sentiment_key, sentiment_result = "Neutral", "Error"
        if sentiment_pipeline is not None:
            try:
                sentiment = sentiment_pipeline(text[:512])[0]
                label = map_sentiment(sentiment["label"])
                sentiment_result = f"{label} ({round(sentiment['score'], 2)})"
                sentiment_key = _extract_sentiment_key(label)
            except Exception as e:
                logger.error("Error de sentimiento en fila %s del lote: %s", idx, e)
        else:
            sentiment_result = "No disponible"
        sentiment_counter[sentiment_key] = sentiment_counter.get(sentiment_key, 0) + 1

        emotion_result = analyze_emotion(text, lang_code) or "-"

        issues = [i for i in extract_issues(text) if i != NO_ISSUES]
        for issue in issues:
            issue_counter[issue] = issue_counter.get(issue, 0) + 1

        is_toxic, _ = analyze_toxicity(text)
        if is_toxic:
            toxic_count += 1

        analyzed.append(
            {
                "texto_completo": text,
                "Texto": text[:120] + ("..." if len(text) > 120 else ""),
                "Idioma": language_display,
                "Sentimiento": sentiment_result,
                "Emocion": emotion_result,
                "Problemas": ", ".join(issues),
                "Toxico": "\u26a0\ufe0f" if is_toxic else "",
            }
        )

    # Segunda pasada: ahora que conocemos la frecuencia global de cada
    # problema en el lote, calculamos la urgencia de cada fila.
    rows = []
    for item in analyzed:
        row_issues = [i for i in item["Problemas"].split(", ") if i]
        max_freq = max((issue_counter.get(i, 0) for i in row_issues), default=0)
        urgency = calculate_urgency(
            item["Sentimiento"], row_issues, frequency=max_freq, is_toxic=(item["Toxico"] != "")
        )
        save_to_history(
            source="batch",
            text=item["texto_completo"],
            language=item["Idioma"],
            sentiment=item["Sentimiento"],
            emotion=item["Emocion"],
            issues=item["Problemas"],
            urgency=urgency,
            toxic=(item["Toxico"] != ""),
        )
        rows.append(
            {
                "Texto": item["Texto"],
                "Idioma": item["Idioma"],
                "Sentimiento": item["Sentimiento"],
                "Emocion": item["Emocion"],
                "Problemas": item["Problemas"] or "-",
                "Urgencia": urgency,
                "Toxico": item["Toxico"],
            }
        )

    results_df = pd.DataFrame(rows)

    total_analizados = len(rows)
    pct = {k: round(100 * v / total_analizados, 1) for k, v in sentiment_counter.items()}
    top_issues = sorted(issue_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    top_issues_text = "\n".join(f"- {i}: {c}" for i, c in top_issues) if top_issues else "- Ninguno detectado"
    predominant_lang = max(lang_counter.items(), key=lambda x: x[1])[0] if lang_counter else "Desconocido"
    urgency_counts = results_df["Urgencia"].value_counts().to_dict() if not results_df.empty else {}

    summary_md = (
        "### Resumen del lote\n\n"
        f"- **Filas en el archivo:** {len(raw_texts)} - **Analizadas:** {total_analizados} - "
        f"**Duplicados casi-identicos excluidos:** {n_duplicates}\n"
        f"- **Sentimiento:** {pct.get('Positivo', 0)}% positivo - "
        f"{pct.get('Neutral', 0)}% neutral - {pct.get('Negativo', 0)}% negativo\n"
        f"- **Idioma predominante:** {predominant_lang}\n"
        f"- **Contenido potencialmente ofensivo:** {toxic_count}/{total_analizados}\n"
        f"- **Urgencia:** Alta={urgency_counts.get(chr(0x1f534) + ' Alta', 0)} - "
        f"Media={urgency_counts.get(chr(0x1f7e1) + ' Media', 0)} - "
        f"Baja={urgency_counts.get(chr(0x1f7e2) + ' Baja', 0)}\n\n"
        f"**Problemas mas frecuentes:**\n{top_issues_text}"
    )

    output_path = None
    try:
        out_dir = os.path.abspath("batch_outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.error("No se pudo escribir el CSV de resultados: %s", e)
        output_path = None

    return summary_md, results_df, output_path


# ============================================================================
# 13) INTERFAZ GRADIO 
# ============================================================================
EXAMPLES_INDIVIDUAL = [
    ["La aplicacion se cierra cada vez que intento subir una foto de perfil, es muy frustrante y ya perdi mi trabajo dos veces."],
    ["No puedo iniciar sesion desde ayer, me sale 'credenciales invalidas' aunque la contrasena es correcta."],
    ["I love the new design, it's so clean and the support team replied within minutes. Great job!"],
    ["The app has become unbearably slow after the last update, everything takes forever to load."],
    ["Esta app es una basura, quien la programo no tiene idea de lo que hace."],
    ["La app funciona bien pero el menu de configuracion podria ser mas claro."],
]

with gr.Blocks(title="AI Feedback Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# \U0001f680 AI Feedback Analyzer\n"
        "Analiza feedback de usuarios automaticamente: idioma, sentimiento, emociones, "
        "problemas, urgencia, toxicidad e historial de tendencias.\n\n"
        "*La primera vez que uses cada funcion, su modelo se descarga y carga en memoria "
        "(puede tardar); las siguientes veces sera mas rapido.*"
    )

    with gr.Tabs():
        # ---------------- TAB 1: analisis individual ----------------
        with gr.TabItem("Analisis individual"):
            input_text = gr.Textbox(
                lines=10,
                label="Feedback",
                placeholder="Pega aqui feedback de usuarios, reviews o comentarios...",
            )
            analyze_btn = gr.Button("Analizar", variant="primary")

            with gr.Row():
                out_lang = gr.Textbox(label="Idioma detectado")
                out_sentiment = gr.Textbox(label="Sentimiento")
                out_emotion = gr.Textbox(label="Emocion")

            out_summary = gr.Textbox(label="Resumen del feedback", lines=3)

            with gr.Row():
                out_issues = gr.Dataframe(headers=["Problema detectado"], label="Problemas detectados", wrap=True)
                out_recs = gr.Dataframe(headers=["Recomendacion"], label="Recomendaciones", wrap=True)

            with gr.Row():
                out_urgency = gr.Textbox(label="Urgencia")
                out_toxicity = gr.Textbox(label="Toxicidad")

            gr.Examples(examples=EXAMPLES_INDIVIDUAL, inputs=input_text, label="Ejemplos")

            analyze_btn.click(
                fn=analyze_feedback,
                inputs=input_text,
                outputs=[out_lang, out_sentiment, out_emotion, out_summary, out_issues, out_recs, out_urgency, out_toxicity],
            )

        # ---------------- TAB 2: analisis por lotes ----------------
        with gr.TabItem("Analisis por lotes (CSV / Excel)"):
            gr.Markdown(
                "Sube un archivo `.csv` o `.xlsx` con una columna de texto "
                "(p. ej. `feedback`, `review` o `comentario`). Se detectan y excluyen "
                "automaticamente las resenas casi-identicas antes de analizar."
            )
            batch_file = gr.File(label="Archivo de resenas", file_types=[".csv", ".xlsx", ".xls"], type="filepath")
            batch_btn = gr.Button("Procesar archivo", variant="primary")
            batch_summary = gr.Markdown()
            batch_results = gr.Dataframe(label="Resultados por fila", wrap=True)
            batch_download = gr.File(label="Descargar resultados (CSV)")

            batch_btn.click(
                fn=process_batch,
                inputs=batch_file,
                outputs=[batch_summary, batch_results, batch_download],
            )

        # ---------------- TAB 3: historial y tendencias ----------------
        with gr.TabItem("Historial y tendencias"):
            gr.Markdown("Cada analisis (individual o por lotes) se guarda automaticamente en SQLite local.")
            refresh_btn = gr.Button("Actualizar historial")
            history_table = gr.Dataframe(label="Ultimos analisis guardados")
            trend_plot = gr.BarPlot(
                label="Tendencia de sentimiento por dia",
                x="fecha",
                y="conteo",
                color="sentimiento_simple",
            )

            def refresh_history():
                return load_history(), history_trend()

            refresh_btn.click(fn=refresh_history, outputs=[history_table, trend_plot])
            demo.load(fn=refresh_history, outputs=[history_table, trend_plot])

if __name__ == "__main__":
    init_db()
    demo.queue().launch()
