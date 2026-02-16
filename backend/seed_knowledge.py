"""
Seed the RAG knowledge base with medical textbook content.

Usage:
    python backend/seed_knowledge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_DIR = REPO_ROOT / "ai-pipeline" / "adaptive-engine"
sys.path.insert(0, str(ADAPTIVE_DIR))

from rag_store import RAGStore  # noqa: E402

DOCUMENTS = [
    # ── Pneumonia ──
    "Pneumonia is an infection of the lungs caused by bacteria, viruses, or fungi. On chest X-rays it typically presents as areas of increased opacity (whiteness) in the lung fields, known as consolidation or infiltrates.",
    "Lobar pneumonia affects an entire lobe and appears as a homogeneous area of consolidation with air bronchograms. It is most commonly caused by Streptococcus pneumoniae.",
    "Bronchopneumonia presents as patchy, multifocal areas of consolidation, often bilateral and predominantly in the lower lobes, affecting the airways and surrounding tissue.",
    "In pediatric patients, viral pneumonia often shows diffuse interstitial patterns (peribronchial thickening, hyperinflation), while bacterial pneumonia shows focal consolidation.",

    # ── Normal CXR ──
    "On a normal chest X-ray the lungs appear as dark (radiolucent) areas because they are filled with air. The heart shadow, mediastinal structures, and bony thorax appear white (radiopaque).",
    "Key anatomical landmarks on a PA chest X-ray include: the trachea (midline), carina (T4-T5 level), aortic knob, hila (containing pulmonary arteries and main bronchi), costophrenic angles, and cardiothoracic ratio.",

    # ── Pathology signs ──
    "The silhouette sign occurs when an intrathoracic opacity is in contact with a border of the heart, aorta, or diaphragm, causing the border to become indistinct. This helps localize the pathology to a specific lobe.",
    "Pleural effusion appears as a meniscus-shaped opacity at the costophrenic angle on an upright chest X-ray. Large effusions can opacify an entire hemithorax.",
    "Atelectasis (lung collapse) results in volume loss and appears as increased opacity with displacement of fissures, mediastinum, or diaphragm toward the affected side.",

    # ── Interpretation approach ──
    "When interpreting a chest X-ray, use a systematic approach: check technical quality (rotation, penetration, inspiration), then review Airway, Bones, Cardiac silhouette, Diaphragm, and lung fields (ABCDE approach).",
    "Common competency gaps in radiology students include: failure to use a systematic approach, satisfaction of search (stopping after finding one abnormality), difficulty distinguishing normal variants from pathology, and incorrect localization of findings.",

    # ── AI/ML context ──
    "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an image most important for a neural network's classification decision. In chest X-ray analysis it typically highlights areas of consolidation, infiltrates, or other pathological findings.",
    "DenseNet121 (used in CheXNet) is a CNN architecture with dense connections between layers. Each layer receives feature maps from all preceding layers, promoting feature reuse and reducing the number of parameters.",
    "CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast by dividing the image into small tiles and applying histogram equalization to each. The clip limit prevents over-amplification of noise.",
    "Image segmentation in medical imaging isolates regions of interest (e.g. lung fields) from the background. U-Net architecture with skip connections is particularly effective for biomedical image segmentation.",
]


def main() -> None:
    kb_dir = str(ADAPTIVE_DIR / "knowledge_base")
    print(f"[INFO] Seeding RAG knowledge base at {kb_dir}")

    store = RAGStore(persist_dir=kb_dir)
    ids = [f"med_doc_{i:03d}" for i in range(len(DOCUMENTS))]
    store.add_documents(DOCUMENTS, ids)

    print(f"[INFO] Added {len(DOCUMENTS)} documents to knowledge base.")

    # Verification query
    results = store.query("What does pneumonia look like on an X-ray?", k=3)
    print(f"[INFO] Verification query returned {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  {i + 1}. {r[:100]}...")


if __name__ == "__main__":
    main()
