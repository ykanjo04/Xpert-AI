
"""
Segmentation pipeline utilities using TensorFlow/Keras.

Features:
- Lightweight residual U-Net with CBAM + ASPP for training.
- tf.data image/mask loaders with optional augmentation and DICOM support.
- Training entrypoint with Dice + BCE loss, checkpoints, and early stopping.
- Inference/eval with optional Hugging Face model loading and mask post-processing.

Usage:
    # Train (assumes image/mask pairs in separate folders)
    python ai-pipeline/segmentation --mode train --images ./data/images --masks ./data/masks

    # Evaluate
    python ai-pipeline/segmentation --mode eval --images ./data/images --masks ./data/masks

    # Inference
    python ai-pipeline/segmentation --mode infer --weights ai-pipeline/models/segmentation_res_cbam_aspp.h5 \
        --image ./data/sample.png --output ./data/sample_mask.png

    # Inference with HF model
    python ai-pipeline/segmentation --mode infer --weights hf://maja011235/lung-segmentation-unet \
        --image ./data/sample.png --output ./data/sample_mask.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Lambda,
    MaxPooling2D,
    Multiply,
    Reshape,
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------------------------------- #
# Metrics & Losses
# --------------------------------------------------------------------------- #

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Compute the Sørensen–Dice coefficient."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss wrapper."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_bce_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Binary cross-entropy plus Dice loss."""
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Intersection-over-Union (IoU) coefficient."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# --------------------------------------------------------------------------- #
# Model (Residual U-Net + CBAM + ASPP)
# --------------------------------------------------------------------------- #

def _residual_block(x: tf.Tensor, filters: int, use_batchnorm: bool, name_prefix: str) -> tf.Tensor:
    shortcut = x
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_conv1")(x)
    if use_batchnorm:
        x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = Activation("relu", name=f"{name_prefix}_act1")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_conv2")(x)
    if use_batchnorm:
        x = BatchNormalization(name=f"{name_prefix}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_proj")(
            shortcut
        )
    x = Add(name=f"{name_prefix}_add")([x, shortcut])
    x = Activation("relu", name=f"{name_prefix}_act2")(x)
    return x


def _cbam_block(x: tf.Tensor, reduction: int, name_prefix: str) -> tf.Tensor:
    channels = x.shape[-1] or 1
    reduced = max(channels // reduction, 1)
    shared_dense_1 = tf.keras.layers.Dense(reduced, activation="relu", name=f"{name_prefix}_mlp1")
    shared_dense_2 = tf.keras.layers.Dense(channels, activation=None, name=f"{name_prefix}_mlp2")

    avg_pool = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    max_pool = GlobalMaxPooling2D(name=f"{name_prefix}_gmp")(x)
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_scale = Activation("sigmoid", name=f"{name_prefix}_ch_sigmoid")(Add()([avg_out, max_out]))
    channel_scale = Reshape((1, 1, channels), name=f"{name_prefix}_ch_reshape")(channel_scale)
    x = Multiply(name=f"{name_prefix}_ch_mul")([x, channel_scale])

    avg_spatial = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name_prefix}_sp_avg")(x)
    max_spatial = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name_prefix}_sp_max")(x)
    spatial = Concatenate(axis=-1, name=f"{name_prefix}_sp_concat")([avg_spatial, max_spatial])
    spatial = Conv2D(1, (7, 7), padding="same", activation="sigmoid", name=f"{name_prefix}_sp_conv")(spatial)
    x = Multiply(name=f"{name_prefix}_sp_mul")([x, spatial])
    return x


def _aspp_block(x: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
    conv_1x1 = Conv2D(filters, (1, 1), padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    conv_3x3_r6 = Conv2D(
        filters, (3, 3), padding="same", dilation_rate=6, activation="relu", name=f"{name_prefix}_conv3_r6"
    )(x)
    conv_3x3_r12 = Conv2D(
        filters, (3, 3), padding="same", dilation_rate=12, activation="relu", name=f"{name_prefix}_conv3_r12"
    )(x)
    conv_3x3_r18 = Conv2D(
        filters, (3, 3), padding="same", dilation_rate=18, activation="relu", name=f"{name_prefix}_conv3_r18"
    )(x)

    pooled = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    pooled = Lambda(lambda t: tf.expand_dims(tf.expand_dims(t, 1), 1), name=f"{name_prefix}_expand")(pooled)
    pooled = Conv2D(filters, (1, 1), padding="same", activation="relu", name=f"{name_prefix}_pool_conv")(pooled)
    target_h = int(x.shape[1]) if x.shape[1] is not None else 1
    target_w = int(x.shape[2]) if x.shape[2] is not None else 1
    pooled = Lambda(
        lambda t: tf.image.resize(t, (target_h, target_w)),
        output_shape=(target_h, target_w, filters),
        name=f"{name_prefix}_resize",
    )(pooled)

    x = Concatenate(name=f"{name_prefix}_concat")([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, pooled])
    x = Conv2D(filters, (1, 1), padding="same", activation="relu", name=f"{name_prefix}_fuse")(x)
    return x


def build_residual_cbam_aspp_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    base_filters: int = 32,
    depth: int = 4,
    num_classes: int = 1,
    use_batchnorm: bool = True,
    dropout_rate: float = 0.1,
    cbam_reduction: int = 8,
) -> Model:
    """
    Build a lightweight residual U-Net with CBAM and ASPP bottleneck.

    Args:
        input_shape: (H, W, C) input dimensions.
        base_filters: Number of filters for the first level.
        depth: Number of down/up-sampling steps.
        num_classes: 1 for binary segmentation; >1 uses softmax.
        use_batchnorm: Enable batch normalization.
        dropout_rate: Dropout applied at each block tail.
        cbam_reduction: Channel attention reduction ratio.
    """
    inputs = Input(shape=input_shape, name="input_image")
    skips = []
    x = inputs

    # Down path
    for d in range(depth):
        filters = base_filters * (2 ** d)
        x = _residual_block(x, filters, use_batchnorm, name_prefix=f"down{d}")
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"down{d}_drop")(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), name=f"pool{d}")(x)

    # Bottleneck + CBAM + ASPP
    x = _residual_block(x, base_filters * (2 ** depth), use_batchnorm, name_prefix="bottleneck")
    x = _cbam_block(x, reduction=cbam_reduction, name_prefix="bottleneck_cbam")
    x = _aspp_block(x, filters=base_filters * (2 ** depth), name_prefix="bottleneck_aspp")

    # Up path
    for d in reversed(range(depth)):
        filters = base_filters * (2 ** d)
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", name=f"up{d}")(x)
        x = Concatenate(name=f"concat{d}")([x, skips[d]])
        x = _residual_block(x, filters, use_batchnorm, name_prefix=f"up{d}")
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"up{d}_drop")(x)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = Conv2D(num_classes, (1, 1), activation=activation, name="mask")(x)
    return Model(inputs=inputs, outputs=outputs, name="res_cbam_aspp_unet")


# --------------------------------------------------------------------------- #
# Data pipeline
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
    # Ensure binary mask
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def _ensure_channels(image: tf.Tensor, channels: int) -> tf.Tensor:
    current = tf.shape(image)[-1]
    if channels == 1:
        return tf.cond(tf.equal(current, 1), lambda: image, lambda: tf.image.rgb_to_grayscale(image))
    if channels == 3:
        return tf.cond(tf.equal(current, 3), lambda: image, lambda: tf.image.grayscale_to_rgb(image))
    return image


def _augment(image: tf.Tensor, mask: tf.Tensor, seed: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simple flip augmentation."""
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    mask = tf.image.random_flip_up_down(mask, seed=seed)
    return image, mask


def _collect_paths(source: str | os.PathLike | Iterable[str], extensions: Optional[Sequence[str]] = None) -> list[str]:
    """Normalize input to a list of file paths."""
    if isinstance(source, (str, os.PathLike)):
        if not extensions:
            extensions = ["png"]
        paths: list[str] = []
        for ext in extensions:
            paths.extend(tf.io.gfile.glob(str(Path(source) / f"*.{ext}")))
        return sorted(paths)
    return sorted([str(p) for p in source])


def create_dataset(
    image_dir: str | os.PathLike | Iterable[str],
    mask_dir: str | os.PathLike | Iterable[str],
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    cache: bool = True,
    seed: int = 42,
    image_exts: Optional[Sequence[str]] = None,
    mask_exts: Optional[Sequence[str]] = None,
    target_channels: int = 3,
) -> tf.data.Dataset:
    """
    Create a tf.data pipeline from image/mask directories.

    Expects matching filenames between image_dir and mask_dir.
    """
    image_paths = _collect_paths(image_dir, extensions=image_exts or ["png", "jpg", "jpeg", "dcm"])
    mask_paths = _collect_paths(mask_dir, extensions=mask_exts or ["png"])
    if len(image_paths) != len(mask_paths):
        raise ValueError("Image and mask counts must match.")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    def _load_pair(img_path: tf.Tensor, msk_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = _load_image(img_path, image_size)
        image = _ensure_channels(image, target_channels)
        mask = _load_mask(msk_path, image_size)
        if augment:
            image, mask = _augment(image, mask, seed)
        return image, mask

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(
    image_dir: str,
    mask_dir: str,
    val_split: float = 0.1,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    base_filters: int = 32,
    depth: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 25,
    weights_path: str = "ai-pipeline/models/segmentation_res_cbam_aspp.h5",
    augment: bool = True,
    input_channels: int = 3,
):
    """Train the residual CBAM + ASPP U-Net model."""
    image_paths = _collect_paths(image_dir, extensions=["png", "jpg", "jpeg", "dcm"])
    mask_paths = _collect_paths(mask_dir, extensions=["png"])
    if len(image_paths) == 0:
        raise ValueError("No images found in the provided directories.")
    if len(image_paths) != len(mask_paths):
        raise ValueError("Image and mask counts must match.")

    split_index = max(1, int(len(image_paths) * (1 - val_split)))
    train_imgs, val_imgs = image_paths[:split_index], image_paths[split_index:]
    train_msks, val_msks = mask_paths[:split_index], mask_paths[split_index:]

    train_ds = create_dataset(
        train_imgs,
        train_msks,
        image_size,
        batch_size,
        shuffle=True,
        augment=augment,
        target_channels=input_channels,
    )
    val_ds = create_dataset(
        val_imgs,
        val_msks,
        image_size,
        batch_size,
        shuffle=False,
        augment=False,
        target_channels=input_channels,
    )

    model = build_residual_cbam_aspp_unet(
        input_shape=(*image_size, input_channels),
        base_filters=base_filters,
        depth=depth,
        num_classes=1,
        use_batchnorm=True,
        dropout_rate=0.1,
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_bce_dice,
        metrics=[dice_coefficient, iou_coefficient, "accuracy"],
    )

    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(weights_path, save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return model, history


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def _is_hf_uri(weights_path: str) -> bool:
    return weights_path.startswith("hf://")


def _infer_input_channels(model: Model, fallback: int = 3) -> int:
    shape = model.input_shape
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if not shape or len(shape) < 4 or shape[-1] is None:
        return fallback
    return int(shape[-1])


def _load_hf_model(weights_path: str) -> Model:
    """Load a HuggingFace-hosted Keras model, supporting both Keras 3 and legacy Keras 2 HDF5."""
    from huggingface_hub import snapshot_download

    # Resolve HF URI to a local snapshot directory
    repo_id = weights_path.replace("hf://", "")
    local_dir = snapshot_download(repo_id=repo_id)

    # Find the model file (.keras or .h5)
    model_file = None
    for fname in sorted(os.listdir(local_dir)):
        if fname.endswith((".keras", ".h5")):
            model_file = os.path.join(local_dir, fname)
            break

    if model_file is None:
        raise FileNotFoundError(f"No .keras or .h5 file found in {local_dir}")

    # Try Keras 3 first
    try:
        import keras
        return keras.saving.load_model(model_file, compile=False)
    except (ValueError, FileNotFoundError, OSError):
        pass

    # Fall back to legacy tf_keras (Keras 2 HDF5 format)
    try:
        import tf_keras
        return tf_keras.models.load_model(model_file, compile=False)
    except ImportError:
        raise ImportError(
            "tf_keras is required to load legacy HDF5 models. "
            "Install with `pip install tf_keras`."
        )


def load_model_for_inference(
    weights_path: str, input_shape: Tuple[int, int, int] = (256, 256, 3), base_filters: int = 32, depth: int = 4
) -> tuple[Model, int, bool]:
    """Load weights for inference, supporting HF model URIs."""
    if _is_hf_uri(weights_path):
        model = _load_hf_model(weights_path)
        return model, _infer_input_channels(model, fallback=input_shape[-1]), True

    model = build_residual_cbam_aspp_unet(
        input_shape=input_shape, base_filters=base_filters, depth=depth, num_classes=1, use_batchnorm=True
    )
    model.load_weights(weights_path)
    return model, input_shape[-1], False


def _dilate(mask: tf.Tensor, kernel_size: int) -> tf.Tensor:
    return tf.nn.max_pool2d(mask, ksize=kernel_size, strides=1, padding="SAME")


def _erode(mask: tf.Tensor, kernel_size: int) -> tf.Tensor:
    return -tf.nn.max_pool2d(-mask, ksize=kernel_size, strides=1, padding="SAME")


def clean_mask(mask: tf.Tensor, kernel_size: int = 5, iterations: int = 1) -> tf.Tensor:
    """Apply a lightweight morphological clean-up (closing + dilation)."""
    x = mask
    for _ in range(iterations):
        x = _dilate(x, kernel_size)
        x = _erode(x, kernel_size)
    x = _dilate(x, kernel_size)
    return tf.clip_by_value(x, 0.0, 1.0)


def predict_mask(
    model: Model,
    image_path: str,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    output_path: Optional[str] = None,
    input_channels: int = 3,
    clean_mask_output: bool = False,
) -> np.ndarray:
    """Run inference on a single image and optionally save the mask."""
    image = _load_image(tf.convert_to_tensor(image_path), image_size)
    image = _ensure_channels(image, input_channels)
    image_np = tf.expand_dims(image, axis=0)
    pred = model.predict(image_np)[0]
    if pred.shape[-1] == 1:
        pred = pred[..., 0]
    mask = (pred > threshold).astype(np.float32)
    if clean_mask_output:
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask_tf = tf.expand_dims(tf.expand_dims(mask_tf, axis=0), axis=-1)
        mask_tf = clean_mask(mask_tf)
        mask = tf.squeeze(mask_tf, axis=[0, -1]).numpy().astype(np.float32)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tf.keras.utils.save_img(out_path, np.expand_dims(mask, axis=-1))
    return mask


def evaluate(
    model: Model,
    image_dir: str | os.PathLike | Iterable[str],
    mask_dir: str | os.PathLike | Iterable[str],
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    threshold: float = 0.5,
    input_channels: int = 3,
    clean_mask_output: bool = False,
) -> dict[str, float]:
    ds = create_dataset(
        image_dir,
        mask_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        cache=False,
        target_channels=input_channels,
    )
    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for images, masks in ds:
        preds = model.predict(images, verbose=0)
        if preds.shape[-1] == 1:
            preds = preds[..., 0]
        preds = (preds > threshold).astype(np.float32)
        masks_np = masks.numpy()
        if masks_np.shape[-1] == 1:
            masks_np = masks_np[..., 0]
        if clean_mask_output:
            cleaned = []
            for pred in preds:
                pred_tf = tf.convert_to_tensor(pred, dtype=tf.float32)
                pred_tf = tf.expand_dims(tf.expand_dims(pred_tf, axis=0), axis=-1)
                pred_tf = clean_mask(pred_tf)
                cleaned.append(tf.squeeze(pred_tf, axis=[0, -1]).numpy())
            preds = np.stack(cleaned, axis=0)

        intersection = np.sum(preds * masks_np, axis=(1, 2))
        pred_sum = np.sum(preds, axis=(1, 2))
        mask_sum = np.sum(masks_np, axis=(1, 2))
        dice = (2.0 * intersection + 1e-6) / (pred_sum + mask_sum + 1e-6)
        union = pred_sum + mask_sum - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice_scores.extend(dice.tolist())
        iou_scores.extend(iou.tolist())

    return {
        "dice_mean": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "dice_median": float(np.median(dice_scores)) if dice_scores else 0.0,
        "iou_mean": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "iou_median": float(np.median(iou_scores)) if iou_scores else 0.0,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual U-Net segmentation pipeline (TensorFlow/Keras).")
    parser.add_argument("--mode", choices=["train", "infer", "eval"], required=True, help="Run training, inference, or evaluation.")
    parser.add_argument("--images", type=str, help="Directory of input images (PNG/JPG/DICOM).")
    parser.add_argument("--masks", type=str, help="Directory of ground-truth masks (PNG).")
    parser.add_argument(
        "--weights",
        type=str,
        default="ai-pipeline/models/segmentation_res_cbam_aspp.h5",
        help="Path to save/load model weights or HF URI.",
    )
    parser.add_argument("--image", type=str, help="Single image path for inference.")
    parser.add_argument("--output", type=str, default="ai-pipeline/models/pred_mask.png", help="Output mask path for inference.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"), help="Image size (height width).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--base_filters", type=int, default=32, help="Base filters for U-Net.")
    parser.add_argument("--depth", type=int, default=4, help="Depth (down/upsampling steps).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--no_augment", action="store_true", help="Disable training augmentation.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask generation.")
    parser.add_argument("--clean_mask", action="store_true", help="Apply morphological cleanup to output masks.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.image_size[0], args.image_size[1])

    if args.mode == "train":
        if not args.images or not args.masks:
            raise SystemExit("Training requires --images and --masks directories.")
        train(
            image_dir=args.images,
            mask_dir=args.masks,
            val_split=args.val_split,
            image_size=image_size,
            batch_size=args.batch_size,
            base_filters=args.base_filters,
            depth=args.depth,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            weights_path=args.weights,
            augment=not args.no_augment,
        )
    elif args.mode == "eval":
        if not args.images or not args.masks:
            raise SystemExit("Evaluation requires --images and --masks directories.")
        model, input_channels, uses_hf = load_model_for_inference(
            args.weights, input_shape=(*image_size, 3), base_filters=args.base_filters, depth=args.depth
        )
        clean_output = args.clean_mask or uses_hf
        metrics = evaluate(
            model,
            args.images,
            args.masks,
            image_size=image_size,
            batch_size=args.batch_size,
            threshold=args.threshold,
            input_channels=input_channels,
            clean_mask_output=clean_output,
        )
        print(
            "Eval metrics: "
            f"dice_mean={metrics['dice_mean']:.4f} "
            f"dice_median={metrics['dice_median']:.4f} "
            f"iou_mean={metrics['iou_mean']:.4f} "
            f"iou_median={metrics['iou_median']:.4f}"
        )
    else:
        if not args.image:
            raise SystemExit("Inference requires --image path.")
        model, input_channels, uses_hf = load_model_for_inference(
            args.weights, input_shape=(*image_size, 3), base_filters=args.base_filters, depth=args.depth
        )
        clean_output = args.clean_mask or uses_hf
        predict_mask(
            model,
            args.image,
            image_size=image_size,
            threshold=args.threshold,
            output_path=args.output,
            input_channels=input_channels,
            clean_mask_output=clean_output,
        )
        print(f"Saved mask to {args.output}")


if __name__ == "__main__":
    main()
