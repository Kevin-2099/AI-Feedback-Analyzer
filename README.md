# 🚀 AI Feedback Analyzer

**AI Feedback Analyzer** es una aplicación basada en Inteligencia Artificial para analizar automáticamente el feedback de usuarios. Permite detectar el idioma, analizar el sentimiento y las emociones, generar resúmenes, identificar problemas frecuentes, calcular la urgencia de cada comentario y procesar grandes volúmenes de reseñas mediante archivos CSV o Excel.

La aplicación está desarrollada con **Gradio** y utiliza modelos de **Hugging Face Transformers**, **pysentimiento**, **Detoxify** y **Sentence Transformers** para ofrecer un análisis completo del feedback.

---

# ✨ Características

## 📝 Análisis individual

Analiza un comentario y obtiene automáticamente:

- 🌍 Detección de idioma
- 😊 Análisis de sentimiento
- 🎭 Detección de emociones
- 📄 Resumen automático
- ⚠️ Identificación de problemas frecuentes
- 💡 Recomendaciones automáticas
- 🚦 Nivel de urgencia
- 🛡️ Detección de lenguaje ofensivo

---

## 📊 Procesamiento por lotes

Analiza archivos **CSV** o **Excel** con cientos o miles de comentarios.

Incluye:

- Procesamiento automático de todas las reseñas
- Detección de idioma
- Sentimiento y emociones
- Problemas detectados
- Nivel de urgencia
- Detección de toxicidad
- Eliminación de comentarios casi duplicados mediante embeddings
- Exportación automática de resultados a CSV

---

## 📈 Historial y tendencias

Todos los análisis se almacenan automáticamente en una base de datos SQLite.

Permite consultar:

- Historial completo de análisis
- Evolución temporal del sentimiento
- Tendencias del feedback
- Estadísticas generales

---

# 🧠 Modelos de IA utilizados

| Función | Tecnología |
|----------|------------|
| Sentimiento | Hugging Face Transformers |
| Resumen automático | Hugging Face Transformers |
| Emociones | pysentimiento |
| Toxicidad | Detoxify |
| Embeddings y deduplicación | Sentence Transformers |
| Detección de idioma | langdetect |

---

# ⚙️ Instalación

Clona el repositorio:

```bash
git clone https://github.com/Kevin-2099/AI-Feedback-Analyzer.git
cd AI-Feedback-Analyzer
```

Crea un entorno virtual:

```bash
python -m venv venv
```

Activación:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
python app.py
```

---

# 🚀 Uso

La interfaz se divide en tres pestañas:

### 📝 Análisis individual

Introduce un comentario y obtén:

- Idioma
- Sentimiento
- Emoción
- Resumen
- Problemas detectados
- Recomendaciones
- Urgencia
- Toxicidad

---

### 📊 Análisis por lotes

Sube un archivo:

- CSV
- XLSX

La aplicación analizará automáticamente todas las reseñas y permitirá descargar un informe con los resultados.

---

### 📈 Historial y tendencias

Consulta:

- Historial almacenado
- Tendencias del sentimiento
- Estadísticas de los análisis realizados

---

# ⚡ Características técnicas

- ✅ Lazy Loading de modelos IA
- ✅ Base de datos SQLite
- ✅ Logging completo
- ✅ Procesamiento por lotes
- ✅ Exportación automática de resultados
- ✅ Detección automática de columnas de texto
- ✅ Gestión robusta de errores
- ✅ Variables de entorno para activar/desactivar funcionalidades
- ✅ Deduplicación mediante embeddings multilingües

---

# 🛠️ Tecnologías

- Python
- Gradio
- Transformers
- Sentence Transformers
- pysentimiento
- Detoxify
- pandas
- SQLite
- langdetect

---

# 📄 Licencia

Este proyecto se distribuye bajo una **licencia propietaria con acceso al código (source-available)**.

El código fuente se pone a disposición únicamente para fines de **visualización, evaluación y aprendizaje**.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un **acuerdo de licencia comercial independiente**.

📌 El texto **legalmente vinculante** de la licencia es la versión en inglés incluida en el archivo `LICENSE`.

Se proporciona una traducción al español en `LICENSE_ES.md` únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.

---

# 👨‍💻 Autor

**Kevin-2099**
