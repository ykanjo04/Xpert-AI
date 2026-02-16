"""
Classification pipeline utilities using TensorFlow/Keras.

Features:
- DenseNet121 (CheXNet-style) backbone with a single sigmoid output.
- Mask-guided classification (ROI masking with lung segmentation mask).
- tf.data CSV loader with optional DICOM support.
- Training/evaluation/inference entrypoints with Grad-CAM.

Usage:
    # Train
    python ai-pipeline/classification --mode train --csv ./data/train.csv --epochs 10

    # Evaluate
    python ai-pipeline/classification --mode eval --csv ./data/val.csv --weights ai-pipeline/models/chexnet_densenet.h5

    # Inference
    python ai-pipeline/classification --mode infer --image ./data/sample.png --mask ./data/sample_mask.png \
        --weights ai-pipeline/models/chexnet_densenet.h5 --output ./outputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #

def _load_standard_image(path: tf.Tensor, image_size: Tuple[int, int], channels: int = 3) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, method="bilinear")
    return image


def _load_dicom(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    def _read_dicom(path_bytes: bytes) -> np.ndarray:
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("pydicom is required to read DICOM files.") from exc

        dicom_path = path_bytes.decode("utf-8")
        ds = pydicom.dcmread(dicom_path)
        array = ds.pixel_array.astype(np.float32)
        if array.size == 0:
            return np.zeros((*image_size, 1), dtype=np.float32)
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val)
        else:
            array = np.zeros_like(array, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., np.newaxis]
        return array

    image = tf.py_function(_read_dicom, [path], Tout=tf.float32)
    image.set_shape([None, None, None])
    image = tf.image.resize(image, image_size, method="bilinear")
    return image


def _load_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    ext = tf.strings.lower(tf.strings.split(path, ".")[-1])
    return tf.cond(
        tf.equal(ext, "dcm"),
        lambda: _load_dicom(path, image_size),
        lambda: _load_standard_image(path, image_size, channels=3),
    )


def _load_mask(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def _ensure_channels(image: tf.Tensor, channels: int) -> tf.Tensor:
    current = tf.shape(image)[-1]
    if channels == 1:
        return tf.cond(tf.equal(current, 1), lambda: image, lambda: tf.image.rgb_to_grayscale(image))
    if channels == 3:
        return tf.cond(tf.equal(current, 3), lambda: image, lambda: tf.image.grayscale_to_rgb(image))
    return image


def _mask_guided_input(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    if image.shape[-1] == 3 and mask.shape[-1] == 1:
        mask = tf.image.grayscale_to_rgb(mask)
    return image * mask


def create_dataset_from_csv(
    csv_path: str | os.PathLike,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    shuffle: bool = True,
    cache: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create a tf.data pipeline from CSV rows.

    CSV columns:
        image_path, mask_path (optional), label (0/1).
    """
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must include image_path and label columns.")
    if "mask_path" not in df.columns:
        df["mask_path"] = ""

    image_paths = df["image_path"].astype(str).tolist()
    mask_paths = df["mask_path"].astype(str).tolist()
    labels = df["label"].astype(np.float32).tolist()

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))

    def _load_row(img_path: tf.Tensor, msk_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = _load_image(img_path, image_size)
        image = _ensure_channels(image, 3)

        mask = tf.cond(
            tf.equal(tf.strings.length(msk_path), 0),
            lambda: tf.ones((image_size[0], image_size[1], 1), dtype=tf.float32),
            lambda: _load_mask(msk_path, image_size),
        )
        image = _mask_guided_input(image, mask)
        return image, tf.expand_dims(label, axis=-1)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_row, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Model and weights
# --------------------------------------------------------------------------- #

def build_densenet121_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
) -> Model:
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        name="densenet121_backbone",
    )
    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(0.2, name="dropout")(x)
    output = Dense(1, activation="sigmoid", name="pneumonia")(x)
    model = Model(inputs=base.input, outputs=output, name="densenet121_pneumonia")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def build_densenet121_14class(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> Model:
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        name="densenet121_backbone",
    )
    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(0.2, name="dropout")(x)
    output = Dense(14, activation="sigmoid", name="chexnet_output")(x)
    return Model(inputs=base.input, outputs=output, name="densenet121_chexnet")


def build_pneumonia_from_chexnet(
    weights_path: str,
    input_shape: Tuple[int, int, int],
    learning_rate: float = 1e-4,
) -> Model:
    """
    Load CheXNet (14-class) weights, then replace the head with a 1-class sigmoid.
    """
    if not weights_path or not os.path.exists(weights_path):
        print(f"[WARN] Weights not found at {weights_path}. Using random initialization.")
        return build_densenet121_classifier(input_shape=input_shape, learning_rate=learning_rate)

    chexnet = build_densenet121_14class(input_shape=input_shape)
    chexnet.load_weights(weights_path, by_name=True, skip_mismatch=True)

    x = chexnet.get_layer("dropout").output
    output = Dense(1, activation="sigmoid", name="pneumonia")(x)
    model = Model(inputs=chexnet.input, outputs=output, name="densenet121_pneumonia")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #

def compute_gradcam(
    model: Model,
    image: tf.Tensor,
    last_conv_layer_name: str = "conv5_block16_concat",
) -> tf.Tensor:
    """
    Compute Grad-CAM heatmap for a single image tensor (H, W, 3).
    Returns a normalized heatmap (H, W).
    """
    if image.ndim == 3:
        image = tf.expand_dims(image, axis=0)

    last_conv = model.get_layer(last_conv_layer_name)
    grad_model = Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[1], image.shape[2]))
    return tf.squeeze(heatmap, axis=-1)


def extract_gradcam_regions(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 80,
) -> list[dict]:
    """Extract bounding-box regions from a Grad-CAM heatmap.

    Returns a list of ``{"x", "y", "w", "h", "intensity"}`` dicts sorted by
    descending intensity.
    """
    import cv2  # local import to avoid top-level dep

    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    binary = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[dict] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < min_area:
            continue
        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "intensity": round(float(np.mean(heatmap[y : y + bh, x : x + bw])), 3),
        })
    return sorted(regions, key=lambda r: r["intensity"], reverse=True)


# --------------------------------------------------------------------------- #
# Training / Evaluation / Inference
# --------------------------------------------------------------------------- #

def train(
    csv_path: str,
    weights_path: str = "ai-pipeline/models/chexnet_densenet.h5",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    output_weights: str = "ai-pipeline/models/densenet121_pneumonia.h5",
):
    model = build_pneumonia_from_chexnet(
        weights_path=weights_path,
        input_shape=(*image_size, 3),
        learning_rate=learning_rate,
    )

    ds = create_dataset_from_csv(csv_path, image_size=image_size, batch_size=batch_size, shuffle=True)

    callbacks = [
        ModelCheckpoint(output_weights, monitor="auc", save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="auc", factor=0.5, patience=3, mode="max", verbose=1),
        EarlyStopping(monitor="auc", patience=5, mode="max", restore_best_weights=True, verbose=1),
    ]

    model.fit(ds, epochs=epochs, callbacks=callbacks)
    return model


def evaluate(
    csv_path: str,
    weights_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
):
    model = build_pneumonia_from_chexnet(
        weights_path=weights_path,
        input_shape=(*image_size, 3),
    )

    ds = create_dataset_from_csv(csv_path, image_size=image_size, batch_size=batch_size, shuffle=False)
    results = model.evaluate(ds, return_dict=True)
    return results


def infer(
    image_path: str,
    mask_path: Optional[str],
    weights_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    threshold_low: float = 0.45,
    threshold_high: float = 0.55,
    preloaded_model: Optional[Model] = None,
):
    model = preloaded_model or build_pneumonia_from_chexnet(
        weights_path=weights_path,
        input_shape=(*image_size, 3),
    )

    image = _load_image(tf.constant(image_path), image_size)
    image = _ensure_channels(image, 3)
    if mask_path:
        mask = _load_mask(tf.constant(mask_path), image_size)
    else:
        mask = tf.ones((image_size[0], image_size[1], 1), dtype=tf.float32)
    image = _mask_guided_input(image, mask)

    pred = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0][0]
    needs_human_review = threshold_low <= pred <= threshold_high

    heatmap = compute_gradcam(model, image)
    heatmap_float = heatmap.numpy()

    # Extract hot regions from heatmap
    gradcam_regions = extract_gradcam_regions(heatmap_float, threshold=0.5)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = output_dir / "gradcam_heatmap.png"
    heatmap_np = (heatmap_float * 255).astype(np.uint8)
    if heatmap_np.ndim == 2:
        heatmap_np = np.expand_dims(heatmap_np, axis=-1)
    tf.keras.utils.save_img(str(heatmap_path), heatmap_np, scale=False)

    result = {
        "pneumonia_score": float(pred),
        "needs_human_review": bool(needs_human_review),
        "heatmap_path": str(heatmap_path),
        "gradcam_regions": gradcam_regions,
    }

    json_path = output_dir / "prediction.json"
    json_path.write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DenseNet121 pneumonia classification")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], required=True)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--mask", type=str, default="")
    parser.add_argument("--weights", type=str, default="ai-pipeline/models/chexnet_densenet.h5")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.mode in {"train", "eval"} and not args.csv:
        raise ValueError("--csv is required for train/eval")
    if args.mode == "infer" and not args.image:
        raise ValueError("--image is required for infer")

    if args.mode == "train":
        train(
            csv_path=args.csv,
            weights_path=args.weights,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
    elif args.mode == "eval":
        results = evaluate(
            csv_path=args.csv,
            weights_path=args.weights,
            batch_size=args.batch_size,
        )
        print(json.dumps(results, indent=2))
    elif args.mode == "infer":
        result = infer(
            image_path=args.image,
            mask_path=args.mask or None,
            weights_path=args.weights,
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
