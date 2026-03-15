import gradio as gr
from transformers import pipeline
from langdetect import detect

# Cargar modelos
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# Mapeo de sentimiento
def map_sentiment(label):
    stars = int(label[0])
    if stars <= 2:
        return "😡 Negativo"
    elif stars == 3:
        return "😐 Neutral"
    else:
        return "😀 Positivo"

# Detección de idioma
def detect_language(text):
    try:
        lang = detect(text)
        if lang == "es":
            return "🇪🇸 Español"
        elif lang == "en":
            return "🇺🇸 Inglés"
        else:
            return f"🌍 {lang}"
    except:
        return "Idioma no detectado"

# Keywords expandidas
issues_keywords = {
    "UI confusa": [
        "confusing interface","hard to navigate","interface not intuitive","poor layout",
        "navigation is confusing","menu is confusing","difficult navigation",
        "interfaz confusa","interfaz poco clara","interfaz no intuitiva",
        "navegación confusa","diseño confuso","menú confuso","interfaz complicada"
    ],
    "Problemas de login": [
        "cannot login","can't login","unable to login","login not working","login failed",
        "authentication failed","invalid credentials","password not accepted","login loop",
        "no puedo iniciar sesión","error al iniciar sesión","credenciales inválidas",
        "fallo de autenticación","problema al iniciar sesión","bucle de login"
    ],
    "Errores o bugs": [
        "app crashes","application crashes","keeps crashing","random crash",
        "crash when opening","unexpected error","internal server error","error message appears",
        "bug in the app","major bug","critical bug","function not working","button not working",
        "la app se cierra","error inesperado","mensaje de error","bug en la aplicación",
        "la app se congela","función no funciona","botón no funciona","falla crítica"
    ],
    "Problemas de rendimiento": [
        "app is slow","very slow app","slow loading","slow performance","laggy interface",
        "takes forever to load","long loading time","performance issues","slow response time",
        "aplicación lenta","muy lenta","tiempo de carga largo","tarda mucho en cargar",
        "rendimiento lento","interfaz lenta","retraso en respuesta","experiencia lenta"
    ],
    "Problemas de soporte": [
        "support not responding","no response from support","support never replied",
        "customer support not helpful","bad customer support","support ignored my message",
        "waiting for support response","slow response from support",
        "soporte no responde","soporte muy lento","no recibí respuesta del soporte",
        "mala atención al cliente","soporte ineficiente","tiempos de respuesta largos"
    ],
    "Problemas de actualización": [
        "update broke the app","update caused problems","after update it crashes",
        "latest update broken","update ruined the app","update introduced bugs",
        "la actualización rompió la app","problema después de actualizar",
        "la última actualización falla","actualización generó errores","actualización inestable"
    ]
}

# 🔎 Detección flexible de problemas por palabras
def extract_issues(text):
    found = []
    lower_text = text.lower()
    for issue, keywords in issues_keywords.items():
        for k in keywords:
            words = k.lower().split()
            if all(word in lower_text for word in words):
                found.append(issue)
                break
    return found if found else ["No se detectaron problemas claros"]

# Generar recomendaciones
def generate_recommendations(issues):
    mapping = {
        "UI confusa": "Mejorar navegación y claridad de la interfaz",
        "Problemas de login": "Revisar sistema de autenticación y flujo de login",
        "Errores o bugs": "Priorizar corrección de errores críticos",
        "Problemas de rendimiento": "Optimizar rendimiento y tiempos de carga",
        "Problemas de soporte": "Mejorar tiempos de respuesta del soporte",
        "Problemas de actualización": "Revisar estabilidad de las últimas actualizaciones"
    }
    recs = []
    for issue in issues:
        if issue in mapping:
            recs.append(mapping[issue])
    if not recs:
        recs.append("Seguir recopilando feedback de usuarios")
    return recs

# Pipeline principal
def analyze_feedback(text):
    if not text.strip():
        return "Sin texto", "", "", "", ""

    # Idioma
    language = detect_language(text)

    # Sentimiento
    sentiment = sentiment_pipeline(text[:512])[0]
    sentiment_label = map_sentiment(sentiment["label"])
    confidence = round(sentiment["score"], 2)
    sentiment_result = f"{sentiment_label} (confianza {confidence})"

    # Resumen
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    except:
        summary = "No se pudo generar resumen."

    # Problemas detectados
    issues = extract_issues(text)
    issues_text = "\n".join([f"• {i}" for i in issues])

    # Recomendaciones
    recs = generate_recommendations(issues)
    recs_text = "\n".join([f"• {r}" for r in recs])

    return language, sentiment_result, summary, issues_text, recs_text

# Interfaz Gradio
demo = gr.Interface(
    fn=analyze_feedback,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Pega aquí feedback de usuarios, reviews o comentarios..."
    ),
    outputs=[
        gr.Textbox(label="Idioma detectado", lines=1),
        gr.Textbox(label="Sentimiento", lines=1),
        gr.Textbox(label="Resumen del feedback", lines=4),
        gr.Textbox(label="⚠️ Problemas detectados", lines=6),
        gr.Textbox(label="💡 Recomendaciones", lines=6)
    ],
    title="🚀 AI Feedback Analyzer",
    description="Analiza feedback de usuarios automáticamente usando IA."
)

demo.launch()
