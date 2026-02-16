# Xpert — AI-Powered Medical Imaging Tutor

Xpert is a full-stack educational platform that helps **medical students** learn chest X-ray interpretation and **doctors** review AI-assisted diagnoses. A single `python` command launches both the React frontend and the FastAPI backend on **localhost:8000**.

---

## Overview

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + Vite + Tailwind CSS (glass-morphism UI) |
| **Backend** | FastAPI + Uvicorn (Python 3.10+) |
| **AI Pipeline** | TensorFlow / Keras, OpenCV, scikit-image, Gemini LLM |
| **Database** | SQLite (sessions, interactions, chat messages) |
| **Knowledge Base** | ChromaDB + RAG (medical textbook retrieval) |

---

## AI Pipeline

Every uploaded X-ray passes through six stages in sequence. The output of each stage feeds into the next:

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  1. Quality  │────▶│ 2. Enhance-  │────▶│ 3. Segmen-   │
 │     Gate     │     │    ment      │     │    tation     │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ 6. Adaptive  │◀────│ 5. Grad-CAM  │◀────│ 4. Classifi- │
 │    Engine    │     │   Heatmap    │     │    cation    │
 └──────────────┘     └──────────────┘     └──────────────┘
```

### Stage Details

| # | Stage | What It Does | Tech |
|---|-------|-------------|------|
| 1 | **Quality Gate** | Measures contrast (std), noise, sharpness (Laplacian variance), and entropy. Scores the image as **Good / Acceptable / Poor** to decide how aggressively to enhance. | OpenCV, scikit-image |
| 2 | **Enhancement** | Selects the best strategy automatically — **CLAHE** for low contrast, **bilateral denoising** for noisy images, or **unsharp masking** for blurry images — then applies it. | OpenCV |
| 3 | **Segmentation** | Isolates the lung fields from the chest X-ray using a **Residual-CBAM-ASPP U-Net**. The segmentation mask is used to focus the classifier on clinically relevant regions only. | TensorFlow/Keras, HuggingFace weights |
| 4 | **Classification** | Feeds the enhanced, masked image into a **CheXNet (DenseNet-121)** model fine-tuned on pneumonia detection. Outputs a pneumonia probability score (0–1). | TensorFlow/Keras |
| 5 | **Grad-CAM** | Computes a **Gradient-weighted Class Activation Map** over the last convolutional layer to visualize which regions drove the prediction. Extracts bounding-box coordinates of the hottest regions. | TensorFlow |
| 6 | **Adaptive Engine** | Takes the CNN prediction, student input, and Grad-CAM regions. Queries the **RAG knowledge base** (ChromaDB) for relevant medical context, then sends everything to **Gemini LLM** to generate a personalized summary, competency-gap analysis, and learning recommendation. | Gemini API, ChromaDB |

### Data Flow

```
User uploads X-ray image
        │
        ▼
  ┌─ Quality Gate ─────────────────────────────────┐
  │  Input:  raw image                             │
  │  Output: quality score + metrics               │
  └────────────────────────┬───────────────────────┘
                           ▼
  ┌─ Enhancement ──────────────────────────────────┐
  │  Input:  raw image + quality metrics           │
  │  Output: enhanced grayscale image              │
  └────────────────────────┬───────────────────────┘
                           ▼
  ┌─ Segmentation ─────────────────────────────────┐
  │  Input:  enhanced image                        │
  │  Output: binary lung mask (256×256)            │
  └────────────────────────┬───────────────────────┘
                           ▼
  ┌─ Classification ───────────────────────────────┐
  │  Input:  enhanced image + lung mask            │
  │  Output: pneumonia score (0–1)                 │
  └────────────────────────┬───────────────────────┘
                           ▼
  ┌─ Grad-CAM ─────────────────────────────────────┐
  │  Input:  model + input image                   │
  │  Output: heatmap image + region bounding boxes │
  └────────────────────────┬───────────────────────┘
                           ▼
  ┌─ Adaptive Engine ──────────────────────────────┐
  │  Input:  prediction + student input + regions  │
  │          + RAG knowledge base context          │
  │  Output: summary, competency gap, recommend.   │
  └────────────────────────┬───────────────────────┘
                           ▼
        JSON response sent to frontend
```

---

## Key Features

- **Conversational AI Chat** — ask questions, request explanations, or type *"quiz me"* for AI-generated knowledge checks.
- **Markdown Rendering** — AI responses are rendered with rich formatting (bold, lists, code blocks, headings).
- **Pipeline Progress Indicator** — animated six-stage progress shown in real time during analysis.
- **Image Comparison Lightbox** — switch between Original / Enhanced / Mask / Grad-CAM views with an overlay mode and opacity slider.
- **AR Visualization** — immersive mode showing skeletal overlay, Grad-CAM hot-spot regions, and live diagnostics driven by real pipeline data.
- **Doctor Dashboard** — view student session reports with accuracy stats, competency gaps, and learning recommendations.
- **Session Persistence** — all interactions and chat messages are stored in SQLite for review.
- **Model Caching** — segmentation and classification models load once at startup; subsequent requests are instant.
- **Quality Gate** — real image-quality assessment replaces the previous stub.
- **Grad-CAM Region Extraction** — bounding boxes around the hottest heatmap regions are returned to the frontend.

---

## Repository Structure

```
Xpert/
├── ai-pipeline/
│   ├── enhancement.py        # Image quality + enhancement
│   ├── segmentation.py       # U-Net lung segmentation
│   ├── classification.py     # CheXNet classification + Grad-CAM
│   ├── adaptive-engine/      # Gemini LLM orchestrator + RAG
│   │   ├── agent_core.py
│   │   ├── rag_store.py
│   │   └── knowledge_base/   # ChromaDB (auto-generated)
│   └── models/               # Model weights (gitignored)
├── backend/
│   ├── run_xpert.py           # FastAPI server + entry point
│   ├── database.py            # SQLite persistence
│   ├── seed_knowledge.py      # Seed RAG knowledge base
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Routing shell
│   │   ├── components/
│   │   │   ├── ui.jsx             # Shared glass-morphism components
│   │   │   ├── LandingPage.jsx    # Login / role selection
│   │   │   ├── ChatInterface.jsx  # Chat + analysis + progress
│   │   │   ├── ImageComparison.jsx# Lightbox viewer
│   │   │   ├── ARMode.jsx         # AR visualization
│   │   │   └── DoctorDashboard.jsx# Student reports
│   │   ├── services/
│   │   │   ├── auth.js        # Client-side auth
│   │   │   └── api.js         # API calls
│   │   └── styles/
│   │       └── customStyles.js
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or later |
| Node.js | 18+ (for frontend build) |
| npm | 9+ |
| Google API key | Set `GOOGLE_API_KEY` env var for Gemini LLM |

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Xpert

# 2. Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Set your Gemini API key
# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your-key-here"
# macOS / Linux
export GOOGLE_API_KEY="your-key-here"

# 5. (Optional) Seed the RAG knowledge base
python backend/seed_knowledge.py

# 6. Run everything (builds frontend + starts server)
python backend/run_xpert.py
```

Open **http://localhost:8000** in your browser. Choose Student or Doctor, enter any OTP, and start analyzing X-rays.

---

## Development Mode

Run frontend and backend separately for hot-reload:

```bash
# Terminal 1 — backend
cd backend
python run_xpert.py

# Terminal 2 — frontend (dev server on :5173, proxied to :8000)
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check + model status |
| `POST` | `/api/analyze` | Upload X-ray → full pipeline → JSON result |
| `POST` | `/api/chat` | Conversational AI with session context |
| `POST` | `/api/sessions` | Create a new session |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/sessions/{id}` | Get session details |
| `GET` | `/api/sessions/{id}/report` | Generate student report |
| `POST` | `/api/sessions/{id}/end` | End a session |
| `GET` | `/api/doctor/students` | List student sessions |
| `GET` | `/api/outputs/{path}` | Serve pipeline output images |

---