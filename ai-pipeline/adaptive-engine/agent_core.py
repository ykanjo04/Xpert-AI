from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from .log_processor import StateObject, process_student_activity
    from .rag_store import RAGStore
except ImportError:  # Allow running as a script from this folder.
    from log_processor import StateObject, process_student_activity
    from rag_store import RAGStore


UNCERTAINTY_MESSAGE = (
    "The AI model has detected high ambiguity in this scan. "
    "Please review the highlighted Grad-CAM regions carefully before proceeding"
)


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required. Install with `pip install google-genai`."
            ) from exc

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""


@dataclass
class OrchestratorConfig:
    model: str = "gemini-2.5-flash"
    rag_top_k: int = 4


class Orchestrator:
    """
    Orchestrator Agent for the MAPE-K loop.
    """

    def __init__(self, rag_store: RAGStore, config: OrchestratorConfig | None = None):
        self._config = config or OrchestratorConfig()
        self._llm = GeminiClient(model=self._config.model)
        self._rag = rag_store

    def monitor(self, log_json: Any) -> StateObject:
        return process_student_activity(log_json)

    def analyze(self, state: StateObject) -> StateObject:
        return state

    def plan(self, state: StateObject) -> Dict[str, Any]:
        query = f"{state.cnn_prediction} {state.competency_gap}"
        context_chunks = self._rag.query(query, k=self._config.rag_top_k)
        prompt = self._build_prompt(state, context_chunks)
        return {"prompt": prompt}

    def execute(self, state: StateObject, plan: Dict[str, Any]) -> str:
        if 0.45 <= state.cnn_score <= 0.55:
            return UNCERTAINTY_MESSAGE
        return self._llm.generate(plan["prompt"])

    def run(self, log_json: Any) -> str:
        state = self.monitor(log_json)
        state = self.analyze(state)
        plan = self.plan(state)
        return self.execute(state, plan)

    def _build_prompt(self, state: StateObject, context_chunks: List[str]) -> str:
        context_text = "\n\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else "None"
        return f"""
You are the Clinical Report Composer. Use ONLY the provided medical context to support claims.

Medical Context:
{context_text}

CNN Summary:
- Prediction: {state.cnn_prediction}
- Score: {state.cnn_score}
- Grad-CAM Regions: {state.gradcam_regions}

Student Interaction:
- Student ID: {state.student_id}
- Student Input: {state.student_input}
- Result: {state.result}
- Competency Gap: {state.competency_gap}

Return JSON only in this schema:
{{
  "summary": "...",
  "competency_gap": "...",
  "learning_recommendation": "...",
  "explanation": "..."
}}
""".strip()
