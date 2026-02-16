from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class StateObject:
    student_id: str
    action: str
    cnn_prediction: str
    cnn_score: float
    student_input: str
    result: str
    gradcam_regions: List[Dict[str, Any]]
    competency_gap: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_json_input(log_json: Any) -> Dict[str, Any]:
    if isinstance(log_json, str):
        return json.loads(log_json)
    if isinstance(log_json, dict):
        return log_json
    raise TypeError("log_json must be a dict or JSON string.")


def _parse_prediction(prediction: Any, score: Any) -> Tuple[str, float]:
    label = ""
    parsed_score = 0.0
    if isinstance(prediction, dict):
        label = str(prediction.get("label") or prediction.get("class") or "").strip()
        score_val = prediction.get("score")
        if score_val is not None:
            parsed_score = float(score_val)
    elif isinstance(prediction, str):
        label = prediction
        if "(" in prediction and ")" in prediction:
            try:
                label = prediction.split("(")[0].strip()
                score_str = prediction.split("(")[-1].replace(")", "").strip()
                parsed_score = float(score_str)
            except ValueError:
                parsed_score = 0.0
    if score is not None:
        try:
            parsed_score = float(score)
        except (TypeError, ValueError):
            pass
    return label.strip(), parsed_score


def _pick_key(payload: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _summarize_gap(
    student_input: str,
    cnn_label: str,
    gradcam_regions: List[Dict[str, Any]],
) -> str:
    if student_input.strip().lower() == cnn_label.strip().lower():
        return "Student aligned with model prediction; no competency gap detected."
    if not gradcam_regions:
        return f"Student labeled as {student_input} while model predicted {cnn_label}; no focal region provided."
    region = max(
        gradcam_regions,
        key=lambda r: float(r.get("w", 0)) * float(r.get("h", 0)),
    )
    x = region.get("x", "unknown")
    y = region.get("y", "unknown")
    return (
        f"Student missed the salient region around x={x}, y={y} while model predicted {cnn_label}."
    )


def process_student_activity(log_json: Any) -> StateObject:
    """
    Parse student interaction logs into a structured state object.

    Expected fields (best-effort):
      - student_id, action, cnn_prediction, cnn_score, student_input, result
      - gradcam_regions: list of {x, y, w, h}
    """
    payload = _parse_json_input(log_json)
    cnn_payload = payload.get("cnn_payload", {})

    student_id = str(_pick_key(payload, ["student_id", "studentId"], "UNKNOWN"))
    action = str(_pick_key(payload, ["action"], "unknown"))
    student_input = str(
        _pick_key(payload, ["student_input", "studentInput", "user_input"], "")
    )
    result = str(_pick_key(payload, ["result", "outcome"], ""))

    prediction_raw = _pick_key(
        payload,
        ["cnn_prediction", "prediction"],
        cnn_payload.get("prediction"),
    )
    score_raw = _pick_key(payload, ["cnn_score", "score"], cnn_payload.get("score"))

    gradcam_regions = _pick_key(
        payload,
        ["gradcam_regions", "gradcam", "grad_cam"],
        cnn_payload.get("gradcam_regions", []),
    )
    if not isinstance(gradcam_regions, list):
        gradcam_regions = []

    cnn_label, cnn_score = _parse_prediction(prediction_raw, score_raw)
    competency_gap = _summarize_gap(student_input, cnn_label, gradcam_regions)

    return StateObject(
        student_id=student_id,
        action=action,
        cnn_prediction=cnn_label,
        cnn_score=cnn_score,
        student_input=student_input,
        result=result,
        gradcam_regions=gradcam_regions,
        competency_gap=competency_gap,
    )
