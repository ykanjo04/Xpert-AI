"""
Xpert AI Pipeline Server
========================

FastAPI application that serves the React frontend and exposes the full
AI pipeline API with:

- Image analysis (enhancement -> segmentation -> classification -> adaptive)
- Conversational AI chat (Gemini LLM with session context)
- Session persistence (SQLite)
- Student reports for doctors
- Model caching (loaded once at startup)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────── #

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_PIPELINE_DIR = REPO_ROOT / "ai-pipeline"
ADAPTIVE_DIR = AI_PIPELINE_DIR / "adaptive-engine"
OUTPUT_DIR = REPO_ROOT / "backend" / "outputs"
UPLOAD_DIR = REPO_ROOT / "backend" / "uploads"

# ── Default config ─────────────────────────────────────────────────────────── #

DEFAULT_SEG_WEIGHTS = "hf://maja011235/lung-segmentation-unet"
DEFAULT_CLS_WEIGHTS = str(AI_PIPELINE_DIR / "models" / "chexnet_densenet.h5")
DEFAULT_LLM_MODEL = "gemini-2.5-flash"

# ── Dynamic module loading ─────────────────────────────────────────────────── #

def _load_module(module_name: str, module_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load AI-pipeline modules at import time
enhancement_mod = _load_module("xpert_enhancement", AI_PIPELINE_DIR / "enhancement.py")
segmentation_mod = _load_module("xpert_segmentation", AI_PIPELINE_DIR / "segmentation.py")
classification_mod = _load_module("xpert_classification", AI_PIPELINE_DIR / "classification.py")

sys.path.insert(0, str(ADAPTIVE_DIR))
agent_core_mod = _load_module("xpert_agent_core", ADAPTIVE_DIR / "agent_core.py")
sys.path.pop(0)

# Local database module
from database import init_db, create_session, add_interaction, add_message  # noqa: E402
from database import get_session, get_session_report, list_sessions  # noqa: E402
from database import get_session_messages, end_session  # noqa: E402

# ── Model cache ────────────────────────────────────────────────────────────── #

_model_cache: dict[str, Any] = {}


def _get_seg_model():
    """Lazy-load and cache the segmentation model."""
    if "seg" not in _model_cache:
        print("[INFO] Loading segmentation model ...")
        try:
            model, channels, uses_hf = segmentation_mod.load_model_for_inference(
                DEFAULT_SEG_WEIGHTS, input_shape=(256, 256, 3), base_filters=32, depth=4,
            )
        except Exception as exc:
            print(f"[WARN] Segmentation weights load failed: {exc}")
            print("[WARN] Falling back to untrained segmentation model.")
            model = segmentation_mod.build_residual_cbam_aspp_unet(
                input_shape=(256, 256, 3), base_filters=32, depth=4,
                num_classes=1, use_batchnorm=True, dropout_rate=0.1,
            )
            channels, uses_hf = 3, False
        _model_cache["seg"] = (model, channels, uses_hf)
        print("[INFO] Segmentation model loaded OK")
    return _model_cache["seg"]


def _get_cls_model():
    """Lazy-load and cache the classification model."""
    if "cls" not in _model_cache:
        print("[INFO] Loading classification model ...")
        model = classification_mod.build_pneumonia_from_chexnet(
            weights_path=DEFAULT_CLS_WEIGHTS, input_shape=(224, 224, 3),
        )
        _model_cache["cls"] = model
        print("[INFO] Classification model loaded OK")
    return _model_cache["cls"]


# ── Pipeline defaults ──────────────────────────────────────────────────────── #

def _default_enhancement_params() -> dict[str, Any]:
    return {
        "clahe_clip": 2.0, "clahe_grid": (8, 8),
        "denoise_method": "bilateral", "gaussian_ksize": 5,
        "bilateral_d": 9, "bilateral_sigma_color": 75, "bilateral_sigma_space": 75,
        "unsharp_amount": 1.2, "unsharp_radius": 3,
    }

def _default_enhancement_thresholds() -> dict[str, float]:
    return {"contrast": 30.0, "noise": 5.0, "blur": 50.0}


def _parse_response(response_text: str) -> dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"message": response_text}


# ── Run pipeline ───────────────────────────────────────────────────────────── #

def run_pipeline(
    image_path: Path,
    output_dir: Path,
    llm_model: str,
    student_id: str,
    student_input: str,
    action: str,
    result: str,
) -> dict[str, Any]:
    """Execute the full AI pipeline using cached models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = output_dir / "enhanced"
    segmentation_dir = output_dir / "segmentation"
    classification_dir = output_dir / "classification"

    # 1. Quality assessment + enhancement
    quality_result = enhancement_mod.predict_image_quality(str(image_path))
    enhanced_result = enhancement_mod.process_image(
        image_path=str(image_path),
        output_dir=str(enhanced_dir),
        thresholds=_default_enhancement_thresholds(),
        params=_default_enhancement_params(),
        verify=False,
    )
    enhanced_path = Path(enhanced_result["output_path"])

    # 2. Segmentation (cached model)
    seg_model, input_channels, uses_hf = _get_seg_model()
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    mask_path = segmentation_dir / "mask.png"
    segmentation_mod.predict_mask(
        model=seg_model,
        image_path=str(enhanced_path),
        image_size=(256, 256),
        threshold=0.5,
        output_path=str(mask_path),
        input_channels=input_channels,
        clean_mask_output=uses_hf,
    )

    # 3. Classification + Grad-CAM (cached model)
    cls_model = _get_cls_model()
    classification_result = classification_mod.infer(
        image_path=str(enhanced_path),
        mask_path=str(mask_path),
        weights_path=DEFAULT_CLS_WEIGHTS,
        output_dir=str(classification_dir),
        preloaded_model=cls_model,
    )

    score = float(classification_result.get("pneumonia_score", 0.0))
    prediction_label = "pneumonia" if score >= 0.5 else "normal"
    gradcam_regions = classification_result.get("gradcam_regions", [])

    # 4. Adaptive engine
    rag_store = agent_core_mod.RAGStore(persist_dir=str(ADAPTIVE_DIR / "knowledge_base"))
    orchestrator_config = agent_core_mod.OrchestratorConfig(model=llm_model)
    orchestrator = agent_core_mod.Orchestrator(rag_store=rag_store, config=orchestrator_config)

    log_json = {
        "student_id": student_id,
        "action": action,
        "student_input": student_input,
        "result": result,
        "cnn_payload": {
            "prediction": prediction_label,
            "score": score,
            "gradcam_regions": gradcam_regions,
        },
    }
    try:
        adaptive_response = orchestrator.run(log_json)
        adaptive_payload = _parse_response(adaptive_response)
    except Exception as exc:
        print(f"[WARN] Adaptive engine failed: {exc}")
        adaptive_payload = {
            "summary": "Adaptive engine unavailable; fallback response.",
            "competency_gap": log_json["cnn_payload"]["prediction"],
            "learning_recommendation": "Review the case details and compare with Grad-CAM output.",
            "explanation": "Fallback response due to LLM API error.",
        }

    adaptive_path = output_dir / "adaptive_response.json"
    adaptive_path.write_text(json.dumps(adaptive_payload, indent=2))

    return {
        "enhanced_image": str(enhanced_path),
        "mask_path": str(mask_path),
        "classification_output": str(classification_dir / "prediction.json"),
        "classification_result": classification_result,
        "adaptive_output": str(adaptive_path),
        "adaptive_payload": adaptive_payload,
        "quality": quality_result,
        "prediction_label": prediction_label,
        "score": score,
        "gradcam_regions": gradcam_regions,
    }


# ── Pydantic models ───────────────────────────────────────────────────────── #

class ChatRequest(BaseModel):
    session_id: str = ""
    message: str
    role: str = "student"

class SessionCreate(BaseModel):
    role: str = "student"
    student_id: str = "anonymous"


# ── FastAPI application ───────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB and preload models.  Shutdown: cleanup."""
    print("[INFO] Initializing database ...")
    init_db()

    print("[INFO] Preloading AI models (first-time may download weights) ...")
    _get_seg_model()
    _get_cls_model()
    print("[INFO] All models loaded OK\n")

    yield  # ── app runs ──

    print("[INFO] Server shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(title="Xpert AI Pipeline", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Health check ──────────────────────────────────────────────────────── #

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "models_loaded": {"segmentation": "seg" in _model_cache, "classification": "cls" in _model_cache},
        }

    # ── Sessions ──────────────────────────────────────────────────────────── #

    @app.post("/api/sessions")
    async def api_create_session(body: SessionCreate):
        sid = create_session(role=body.role, student_id=body.student_id)
        return {"session_id": sid}

    @app.get("/api/sessions")
    async def api_list_sessions(role: Optional[str] = None):
        return {"sessions": list_sessions(role=role)}

    @app.get("/api/sessions/{session_id}")
    async def api_get_session(session_id: str):
        s = get_session(session_id)
        if not s:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        return s

    @app.get("/api/sessions/{session_id}/report")
    async def api_session_report(session_id: str):
        report = get_session_report(session_id)
        if not report:
            return JSONResponse(status_code=404, content={"error": "Session not found"})
        return report

    @app.post("/api/sessions/{session_id}/end")
    async def api_end_session(session_id: str):
        end_session(session_id)
        return {"status": "ended"}

    @app.get("/api/doctor/students")
    async def api_doctor_students():
        sessions = list_sessions(role="student")
        return {"sessions": sessions}

    # ── Image analysis ────────────────────────────────────────────────────── #

    @app.post("/api/analyze")
    async def analyze(
        file: UploadFile = File(...),
        student_id: str = Form("S001"),
        student_input: str = Form("normal"),
        action: str = Form("diagnose"),
        result: str = Form("pending"),
        session_id: str = Form(""),
    ):
        """Accept an uploaded X-ray, run the full AI pipeline, return results."""
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = UPLOAD_DIR / unique_name
        with open(upload_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        try:
            pipeline_out = run_pipeline(
                image_path=upload_path,
                output_dir=OUTPUT_DIR,
                llm_model=DEFAULT_LLM_MODEL,
                student_id=student_id,
                student_input=student_input,
                action=action,
                result=result,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {exc}"})

        # Build image URLs
        def _to_url(abs_path_str: str) -> str | None:
            p = Path(abs_path_str)
            if not p.exists():
                return None
            try:
                return f"/api/outputs/{p.relative_to(OUTPUT_DIR).as_posix()}"
            except ValueError:
                return None

        enhanced_url = _to_url(pipeline_out["enhanced_image"])
        mask_url = _to_url(pipeline_out["mask_path"])
        gradcam_path = Path(pipeline_out["classification_output"]).parent / "gradcam_heatmap.png"
        gradcam_url = _to_url(str(gradcam_path))

        # Parse adaptive response
        adaptive = pipeline_out["adaptive_payload"]
        parsed_adaptive = adaptive
        if isinstance(adaptive.get("message"), str):
            try:
                cleaned = adaptive["message"].replace("```json\n", "").replace("```", "").strip()
                parsed_adaptive = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                parsed_adaptive = adaptive

        adaptive_summary = parsed_adaptive.get("summary") or parsed_adaptive.get("message", "")
        adaptive_recommendation = parsed_adaptive.get("learning_recommendation") or parsed_adaptive.get("explanation", "")
        competency_gap = parsed_adaptive.get("competency_gap", "")

        # Persist interaction
        if session_id:
            add_interaction(
                session_id,
                image_filename=unique_name,
                prediction=pipeline_out["prediction_label"],
                pneumonia_score=pipeline_out["score"],
                student_input=student_input,
                competency_gap=competency_gap,
                adaptive_summary=adaptive_summary,
                adaptive_recommendation=adaptive_recommendation,
                quality_label=pipeline_out["quality"].get("quality_label"),
                quality_score=pipeline_out["quality"].get("quality_score"),
                gradcam_regions=pipeline_out["gradcam_regions"],
            )

        quality = pipeline_out["quality"]

        return {
            "prediction": pipeline_out["prediction_label"],
            "pneumonia_score": pipeline_out["score"],
            "needs_human_review": pipeline_out["classification_result"].get("needs_human_review", False),
            "quality": {
                "label": quality.get("quality_label", "P"),
                "score": quality.get("quality_score", 0),
            },
            "adaptive": parsed_adaptive,
            "gradcam_regions": pipeline_out["gradcam_regions"],
            "images": {
                "enhanced": enhanced_url,
                "mask": mask_url,
                "gradcam": gradcam_url,
            },
        }

    # ── Conversational chat ───────────────────────────────────────────────── #

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        """Send a text message to the AI tutor with conversational context."""
        # Get conversation history for context
        history = []
        if body.session_id:
            history = get_session_messages(body.session_id, limit=20)

        context_lines = []
        for msg in history[-10:]:
            sender = "Student" if msg["sender"] == "user" else "Xpert AI"
            context_lines.append(f"{sender}: {msg['content']}")

        context_text = "\n".join(context_lines) if context_lines else "(No previous messages)"

        prompt = f"""You are Xpert, an AI medical imaging tutor specializing in chest X-ray
interpretation and pneumonia detection. You are helping a {body.role}.

Previous conversation:
{context_text}

{body.role.capitalize()}: {body.message}

Instructions:
- Be educational, encouraging, and precise.
- Use markdown formatting for clarity (bold, lists, headers).
- If the user asks about a previous analysis, reference the conversation history.
- If the user asks for a quiz, generate 2-3 multiple-choice questions about chest X-ray interpretation.
- Keep responses concise but informative.
- For students: explain concepts, guide learning, ask follow-up questions.
- For doctors: be clinical, reference evidence, provide differential diagnoses.

Respond now:"""

        try:
            llm = agent_core_mod.GeminiClient(model=DEFAULT_LLM_MODEL)
            response_text = llm.generate(prompt)
        except Exception as exc:
            print(f"[WARN] Chat LLM failed: {exc}")
            response_text = (
                "I'm having trouble connecting to the AI service right now. "
                "Please try again, or upload an X-ray for automated analysis."
            )

        # Persist messages
        if body.session_id:
            add_message(body.session_id, "user", body.message)
            add_message(body.session_id, "ai", response_text)

        return {"response": response_text}

    # ── Serve pipeline output files ───────────────────────────────────────── #

    @app.get("/api/outputs/{file_path:path}")
    async def serve_output(file_path: str):
        full_path = OUTPUT_DIR / file_path
        if not full_path.exists() or not full_path.is_file():
            return JSONResponse(status_code=404, content={"error": "File not found"})
        return FileResponse(full_path)

    # ── Serve the built frontend ──────────────────────────────────────────── #

    frontend_dist = REPO_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            candidate = frontend_dist / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")
    else:
        @app.get("/")
        async def no_frontend():
            return JSONResponse(content={"message": "Frontend not built. Run the server to auto-build."})

    return app


# ── Build frontend ─────────────────────────────────────────────────────────── #

def _build_frontend() -> None:
    frontend_dir = REPO_ROOT / "frontend"
    if not frontend_dir.exists():
        print("[WARN] frontend/ directory not found - skipping build.")
        return

    dist_dir = frontend_dir / "dist"
    if dist_dir.exists():
        print("[INFO] frontend/dist already exists - skipping build. Delete it to force a rebuild.")
        return

    npm_cmd = shutil.which("npm")
    if npm_cmd is None:
        print("[WARN] npm not found on PATH - cannot build frontend.")
        return

    print("[INFO] Installing frontend dependencies ...")
    subprocess.run([npm_cmd, "install"], cwd=str(frontend_dir), check=True)
    print("[INFO] Building frontend (vite build) ...")
    subprocess.run([npm_cmd, "run", "build"], cwd=str(frontend_dir), check=True)
    print("[INFO] Frontend built successfully.")


# ── Entry point ────────────────────────────────────────────────────────────── #

def main() -> None:
    print("=" * 60)
    print("  Xpert AI Pipeline Server  v2.0")
    print("=" * 60)

    _build_frontend()

    print("\n[INFO] Starting server on http://localhost:8000")
    print("[INFO] Press Ctrl+C to stop.\n")

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
