import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import laplace
from skimage.measure import shannon_entropy

def _load_grayscale(image_path: str, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return cv2.resize(img, size)


def _compute_metrics(img: np.ndarray) -> dict[str, float]:
    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    lap_var = float(np.var(laplace(img)))
    entropy = float(shannon_entropy(img))
    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)
    noise = float(np.mean(np.abs(img - cv2.GaussianBlur(img, (5, 5), 0))))
    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "lap_var": lap_var,
        "entropy": entropy,
        "edge_density": edge_density,
        "noise": noise,
    }


def extract_quality_features(image_path: str) -> np.ndarray:
    img = _load_grayscale(image_path)
    metrics = _compute_metrics(img)
    return np.array(
        [
            metrics["mean_intensity"],
            metrics["std_intensity"],
            metrics["lap_var"],
            metrics["entropy"],
            metrics["edge_density"],
            metrics["noise"],
        ]
    ).reshape(1, -1)

def predict_image_quality(image_path: str) -> dict[str, float | str]:
    """Assess image quality using computed metrics.

    Quality labels:
        G — Good   (score >= 0.6)
        A — Acceptable (0.4 <= score < 0.6)
        P — Poor   (score < 0.4)
    """
    try:
        img = _load_grayscale(image_path)
        metrics = _compute_metrics(img)

        score = 0.0
        # Contrast: higher std ≈ better contrast
        score += min(metrics["std_intensity"] / 50.0, 1.0) * 0.3
        # Noise: lower is better
        score += max(0.0, 1.0 - metrics["noise"] / 10.0) * 0.3
        # Sharpness: higher Laplacian variance ≈ sharper
        score += min(metrics["lap_var"] / 100.0, 1.0) * 0.2
        # Entropy: sweet-spot around 5.5
        entropy_score = 1.0 - abs(metrics["entropy"] - 5.5) / 3.0
        score += max(0.0, entropy_score) * 0.2

        label = "G" if score >= 0.6 else ("A" if score >= 0.4 else "P")
        return {
            "quality_label": label,
            "quality_score": round(score, 3),
            "metrics": metrics,
        }
    except Exception:
        return {"quality_label": "P", "quality_score": 0.0}


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def apply_denoise(
    img: np.ndarray,
    method: str = "bilateral",
    gaussian_ksize: int = 5,
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 75,
    bilateral_sigma_space: int = 75,
) -> np.ndarray:
    if method == "gaussian":
        ksize = (gaussian_ksize, gaussian_ksize)
        return cv2.GaussianBlur(img, ksize, 0)
    return cv2.bilateralFilter(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)


def apply_unsharp_mask(img: np.ndarray, amount: float = 1.2, radius: int = 3) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def select_enhancement(metrics: dict[str, float], thresholds: dict[str, float]) -> str:
    contrast_deficit = max(0.0, (thresholds["contrast"] - metrics["std_intensity"]) / thresholds["contrast"])
    noise_excess = max(0.0, (metrics["noise"] - thresholds["noise"]) / thresholds["noise"])
    blur_deficit = max(0.0, (thresholds["blur"] - metrics["lap_var"]) / thresholds["blur"])

    scores = {
        "clahe": contrast_deficit,
        "denoise": noise_excess,
        "unsharp": blur_deficit,
    }
    return max(scores, key=scores.get)


def process_image(
    image_path: str,
    output_dir: str,
    thresholds: dict[str, float],
    params: dict[str, float | int | str | tuple[int, int]],
    verify: bool = True,
) -> dict[str, object]:
    quality = predict_image_quality(image_path)
    img = _load_grayscale(image_path)
    metrics = _compute_metrics(img)
    choice = select_enhancement(metrics, thresholds)

    if choice == "clahe":
        enhanced = apply_clahe(
            img,
            clip_limit=float(params["clahe_clip"]),
            tile_grid=tuple(params["clahe_grid"]),
        )
    elif choice == "denoise":
        enhanced = apply_denoise(
            img,
            method=str(params["denoise_method"]),
            gaussian_ksize=int(params["gaussian_ksize"]),
            bilateral_d=int(params["bilateral_d"]),
            bilateral_sigma_color=int(params["bilateral_sigma_color"]),
            bilateral_sigma_space=int(params["bilateral_sigma_space"]),
        )
    else:
        enhanced = apply_unsharp_mask(
            img,
            amount=float(params["unsharp_amount"]),
            radius=int(params["unsharp_radius"]),
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir_path / f"enhanced_{Path(image_path).stem}.png")
    cv2.imwrite(output_path, enhanced)
    applied = choice

    post_quality = predict_image_quality(output_path) if verify else {}
    return {
        "input_path": image_path,
        "output_path": output_path,
        "quality": quality,
        "applied": applied,
        "post_quality": post_quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image quality enhancement for X-ray preprocessing.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--output_dir", type=str, default="ai-pipeline/models/enhanced", help="Output directory.")
    parser.add_argument("--contrast_thresh", type=float, default=30.0, help="STD threshold for low contrast.")
    parser.add_argument("--noise_thresh", type=float, default=5.0, help="Noise threshold for denoising.")
    parser.add_argument("--blur_thresh", type=float, default=50.0, help="Laplacian variance threshold for blur.")
    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe_grid", type=int, nargs=2, default=[8, 8], metavar=("GX", "GY"), help="CLAHE tile grid.")
    parser.add_argument("--denoise_method", type=str, default="bilateral", choices=["bilateral", "gaussian"])
    parser.add_argument("--gaussian_ksize", type=int, default=5, help="Gaussian blur kernel size.")
    parser.add_argument("--bilateral_d", type=int, default=9, help="Bilateral filter diameter.")
    parser.add_argument("--bilateral_sigma_color", type=int, default=75, help="Bilateral sigmaColor.")
    parser.add_argument("--bilateral_sigma_space", type=int, default=75, help="Bilateral sigmaSpace.")
    parser.add_argument("--unsharp_amount", type=float, default=1.2, help="Unsharp mask amount.")
    parser.add_argument("--unsharp_radius", type=int, default=3, help="Unsharp mask radius.")
    parser.add_argument("--no_verify", action="store_true", help="Skip re-checking quality after enhancement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {
        "contrast": float(args.contrast_thresh),
        "noise": float(args.noise_thresh),
        "blur": float(args.blur_thresh),
    }
    params = {
        "clahe_clip": args.clahe_clip,
        "clahe_grid": (args.clahe_grid[0], args.clahe_grid[1]),
        "denoise_method": args.denoise_method,
        "gaussian_ksize": args.gaussian_ksize,
        "bilateral_d": args.bilateral_d,
        "bilateral_sigma_color": args.bilateral_sigma_color,
        "bilateral_sigma_space": args.bilateral_sigma_space,
        "unsharp_amount": args.unsharp_amount,
        "unsharp_radius": args.unsharp_radius,
    }

    result = process_image(
        args.image,
        output_dir=args.output_dir,
        thresholds=thresholds,
        params=params,
        verify=not args.no_verify,
    )
    summary_path = Path(args.output_dir) / "enhancement_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()