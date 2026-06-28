# 📄 CV Analyzer

An AI-powered CV analysis web application built with Flask, spaCy NLP, and pdfplumber. Upload a PDF resume and instantly extract key information, analyse skills, and visualise insights — all through a clean web interface. Fully containerised with Docker for easy deployment.

---

## ✨ Features

- **PDF Resume Parsing** — Extracts raw text from uploaded PDF CVs using `pdfplumber`
- **NLP-Powered Analysis** — Processes resume content using spaCy's `en_core_web_md` model for named entity recognition and keyword extraction
- **Skills & Entity Extraction** — Identifies candidate skills, experience, education, and key entities from unstructured text
- **Data Visualisation** — Generates charts and visual summaries using `matplotlib`
- **Web Interface** — Clean HTML/CSS frontend served by Flask for uploading and viewing results
- **Docker Support** — Fully containerised for one-command deployment anywhere

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask 2.0+ |
| NLP Engine | spaCy (`en_core_web_md`) |
| PDF Parsing | pdfplumber |
| Visualisation | matplotlib |
| Frontend | HTML, CSS |
| Containerisation | Docker |

---

## 📁 Project Structure

```
CV-Analyzer/
├── app/                  # Flask application package
│   ├── __init__.py       # App factory (create_app)
│   ├── routes/           # URL routes & view logic
│   ├── templates/        # HTML templates
│   └── static/           # CSS & static assets
├── Dockerfile            # Docker container definition
├── req.txt               # Python dependencies
└── run.py                # Application entry point
```

---

## 🚀 Getting Started

### Option A — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/malikKanishka/CV-Analyzer.git
cd CV-Analyzer
```

**2. Install dependencies**
```bash
pip install -r req.txt
python -m spacy download en_core_web_md
```

**3. Run the app**
```bash
python run.py
```

Open your browser at:
```
http://localhost:5000
```

---

### Option B — Run with Docker

**1. Build the image**
```bash
docker build -t cv-analyzer .
```

**2. Run the container**
```bash
docker run -p 5000:5000 cv-analyzer
```

Open your browser at:
```
http://localhost:5000
```

---

## 📦 Dependencies

```
Flask>=2.0
spacy>=3.0
pdfplumber>=0.5
matplotlib>=3.0
```

Install with:
```bash
pip install -r req.txt
python -m spacy download en_core_web_md
```

---

## 🔄 How It Works

```
Upload PDF → pdfplumber extracts text → spaCy NLP analyses content
→ Skills & entities extracted → matplotlib generates charts → Results displayed in browser
```

1. User uploads a PDF resume via the web interface
2. `pdfplumber` reads and extracts raw text from the PDF
3. spaCy's `en_core_web_md` model processes the text and identifies entities, skills, and key information
4. `matplotlib` generates visual summaries of the extracted data
5. Results are rendered back to the user in the browser

---

## 🐳 Docker Details

The Dockerfile:
- Uses `python:3.10` as the base image
- Sets working directory to `/ATS`
- Installs all dependencies from `req.txt`
- Downloads the spaCy `en_core_web_md` language model
- Exposes port `5000`
- Starts the app via `python run.py`

---
