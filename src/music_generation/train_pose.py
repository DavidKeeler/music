"""Pose-conditioned training script for mel generator."""
import argparse
import logging
import math
import tensorflow as tf
import keras
from pathlib import Path

from .config import (
    DATA_DIR, CACHE_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    INITIAL_TF_RATIO, MIN_TF_RATIO, TF_DECAY_K, TF_WARMUP_STEPS,
    N_MELS, KL_BETA, POSE_FEATURE_DIM,
)
from .dataset import create_pose_dataset
from .model import MelGenerator, MelTokenizer, MelDetokenizer, LatentEncoder
from .layers import CausalConv1D, CausalConvBlock, LocalWindowAttention, TransformerBlock
from .train import (
    configure_memory, check_batch_size, WarmupCosineSchedule,
    exponential_tf_schedule, MelGeneratorTraining,
)

logger = logging.getLogger(__name__)


def cosine_alpha_schedule(step, alpha_start, alpha_end, total_steps):
    """Cosine ramp from alpha_start to alpha_end over total_steps."""
    progress = tf.minimum(tf.cast(step, tf.float32) / tf.maximum(tf.cast(total_steps, tf.float32), 1.0), 1.0)
    return alpha_start + (alpha_end - alpha_start) * 0.5 * (1.0 - tf.cos(math.pi * progress))


def apply_conditioning_dropout(pose_embedded, dropout_rate):
    """Zero out pose for random samples. Returns masked pose."""
    if dropout_rate <= 0.0:
        return pose_embedded
    batch_size = tf.shape(pose_embedded)[0]
    mask = tf.cast(tf.random.uniform([batch_size, 1, 1]) >= dropout_rate, pose_embedded.dtype)
    return pose_embedded * mask


@keras.saving.register_keras_serializable()
class PoseConditionedTraining(tf.keras.Model):
    """Training wrapper for pose-conditioned mel generation."""

    def __init__(self, base_model, alpha_start=0.0, alpha_end=0.3, alpha_steps=100000,
                 cond_dropout_rate=0.0, initial_tf_ratio=INITIAL_TF_RATIO,
                 min_tf_ratio=MIN_TF_RATIO, decay_k=TF_DECAY_K,
                 warmup_steps=TF_WARMUP_STEPS, kl_beta=KL_BETA, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.latent_encoder = LatentEncoder()

        # Alpha schedule params
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_steps = alpha_steps
        self.cond_dropout_rate = cond_dropout_rate

        # Teacher forcing params
        self.initial_tf_ratio = initial_tf_ratio
        self.min_tf_ratio = min_tf_ratio
        self.decay_k = decay_k
        self.warmup_steps = warmup_steps
        self.kl_beta = kl_beta

        # Tracking variables
        self.tf_ratio = tf.Variable(initial_tf_ratio, trainable=False, dtype=tf.float32, name="tf_ratio")
        self.training_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="training_step")

        # Metrics
        self.mel_loss_metric = tf.keras.metrics.Mean(name="mel_loss")
        self.kl_loss_metric = tf.keras.metrics.Mean(name="kl_loss")
        self.tf_ratio_metric = tf.keras.metrics.Mean(name="tf_ratio")
        self.alpha_metric = tf.keras.metrics.Mean(name="alpha")

    @property
    def metrics(self):
        return [self.mel_loss_metric, self.kl_loss_metric, self.tf_ratio_metric, self.alpha_metric]

    def _update_schedules(self):
        """Update alpha and teacher forcing ratio."""
        alpha = cosine_alpha_schedule(self.training_step, self.alpha_start, self.alpha_end, self.alpha_steps)
        self.base_model.pose_alpha.assign(alpha)

        step_after_warmup = tf.maximum(tf.cast(0, tf.int64), self.training_step - tf.cast(self.warmup_steps, tf.int64))
        new_ratio = exponential_tf_schedule(step_after_warmup, self.initial_tf_ratio, self.min_tf_ratio, self.decay_k)
        self.tf_ratio.assign(new_ratio)
        self.training_step.assign_add(1)

    def call(self, inputs, training=False):
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            x, _, pose = inputs
        else:
            x, pose = inputs, None
        z, _, _ = self.latent_encoder(x, training=training)
        return self.base_model(x, z=z, pose=pose, training=training)

    def _compute_kl_loss(self, z_mean, z_logvar):
        return -0.5 * tf.reduce_mean(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))

    def _encode_pose(self, pose):
        """Encode and apply conditioning dropout to pose."""
        pose_embedded = self.base_model.pose_encoder(pose, training=True)
        pose_embedded = self.base_model.pose_proj(pose_embedded)
        pose_embedded = self.base_model.pose_stride_conv(pose_embedded)
        return apply_conditioning_dropout(pose_embedded, self.cond_dropout_rate)

    def _pure_teacher_forcing(self, x, y, pose_embedded):
        with tf.GradientTape() as tape:
            z, z_mean, z_logvar = self.latent_encoder(x, training=True)
            tokens = self.base_model.tokenizer(x, training=True)
            token_preds = self.base_model.forward_tokens(tokens, z=z, pose_embedded=pose_embedded, training=True)
            preds = self.base_model.detokenizer(token_preds, target_length=tf.shape(x)[1], training=True)

            mel_loss = tf.reduce_mean(tf.square(preds - y))
            kl_loss = self._compute_kl_loss(z_mean, z_logvar)
            loss = mel_loss + self.kl_beta * kl_loss

        trainable_vars = self.base_model.trainable_variables + self.latent_encoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self._update_metrics(mel_loss, kl_loss)
        return {"loss": loss, "mel_loss": self.mel_loss_metric.result(), "kl_loss": self.kl_loss_metric.result(),
                "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result(), "alpha": self.alpha_metric.result()}

    def _parallel_scheduled_sampling(self, x, y, pose_embedded):
        # Pass 1: predictions (detached)
        z_pass1, _, _ = self.latent_encoder(x, training=False)
        gt_tokens_detached = self.base_model.tokenizer(x, training=False)
        pred_tokens = self.base_model.forward_tokens(gt_tokens_detached, z=z_pass1, pose_embedded=pose_embedded, training=False)

        pred_tokens_shifted = tf.concat([gt_tokens_detached[:, :1, :], pred_tokens[:, :-1, :]], axis=1)
        tok_seq_len = tf.shape(gt_tokens_detached)[1]
        use_teacher = tf.random.uniform([tf.shape(x)[0], tok_seq_len, 1]) < self.tf_ratio

        # Pass 2: train
        with tf.GradientTape() as tape:
            z, z_mean, z_logvar = self.latent_encoder(x, training=True)
            gt_tokens = self.base_model.tokenizer(x, training=True)
            mixed_tokens = tf.where(use_teacher, gt_tokens, tf.stop_gradient(pred_tokens_shifted))
            preds = self.base_model.forward_from_tokens(mixed_tokens, z=z, pose_embedded=pose_embedded, training=True)

            mel_loss = tf.reduce_mean(tf.square(preds - y))
            kl_loss = self._compute_kl_loss(z_mean, z_logvar)
            loss = mel_loss + self.kl_beta * kl_loss

        trainable_vars = self.base_model.trainable_variables + self.latent_encoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self._update_metrics(mel_loss, kl_loss)
        return {"loss": loss, "mel_loss": self.mel_loss_metric.result(), "kl_loss": self.kl_loss_metric.result(),
                "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result(), "alpha": self.alpha_metric.result()}

    def _update_metrics(self, mel_loss, kl_loss):
        self.mel_loss_metric.update_state(mel_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.tf_ratio_metric.update_state(self.tf_ratio)
        self.alpha_metric.update_state(self.base_model.pose_alpha)

    def train_step(self, data):
        self._update_schedules()
        x, y, pose = data
        pose_embedded = self._encode_pose(pose)

        return tf.cond(
            self.tf_ratio >= 0.99,
            lambda: self._pure_teacher_forcing(x, y, pose_embedded),
            lambda: self._parallel_scheduled_sampling(x, y, pose_embedded),
        )

    def get_config(self):
        return {
            "base_model": tf.keras.utils.serialize_keras_object(self.base_model),
            "alpha_start": float(self.alpha_start),
            "alpha_end": float(self.alpha_end),
            "alpha_steps": int(self.alpha_steps),
            "cond_dropout_rate": float(self.cond_dropout_rate),
            "initial_tf_ratio": float(self.initial_tf_ratio),
            "min_tf_ratio": float(self.min_tf_ratio),
            "decay_k": float(self.decay_k),
            "warmup_steps": int(self.warmup_steps),
            "kl_beta": float(self.kl_beta),
        }

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.utils.deserialize_keras_object(config["base_model"])
        return cls(base_model, **{k: v for k, v in config.items() if k != "base_model"})


def load_checkpoint_with_autodetect(model, checkpoint_path):
    """Load checkpoint, auto-initializing missing pose layers for Phase 1 checkpoints."""
    checkpoint_path = str(checkpoint_path)
    custom_objs = {
        'PoseConditionedTraining': PoseConditionedTraining,
        'MelGeneratorTraining': MelGeneratorTraining,
        'WarmupCosineSchedule': WarmupCosineSchedule,
        'MelGenerator': MelGenerator,
        'MelTokenizer': MelTokenizer,
        'MelDetokenizer': MelDetokenizer,
        'LatentEncoder': LatentEncoder,
        'CausalConv1D': CausalConv1D,
        'CausalConvBlock': CausalConvBlock,
        'LocalWindowAttention': LocalWindowAttention,
        'TransformerBlock': TransformerBlock,
    }

    # Try loading as a full pose-conditioned checkpoint first
    try:
        with tf.keras.utils.custom_object_scope(custom_objs):
            loaded = tf.keras.models.load_model(checkpoint_path)
        if isinstance(loaded, PoseConditionedTraining):
            logger.info("Loaded Phase 2/3 pose-conditioned checkpoint")
            return loaded
    except Exception:
        pass

    # Try loading as Phase 1 (audio-only) full model
    logger.info("Attempting Phase 1 (audio-only) checkpoint load with partial weight matching")
    try:
        with tf.keras.utils.custom_object_scope(custom_objs):
            loaded = tf.keras.models.load_model(checkpoint_path)
        source = loaded.base_model if hasattr(loaded, 'base_model') else loaded
        _copy_matching_weights(source, model.base_model)
        logger.info("Phase 1 checkpoint loaded — new pose layers initialized from scratch")
        return model
    except Exception as e:
        logger.warning(f"Full model load failed ({e}), trying weights-only")

    # Fallback: build a temporary Phase 1 model, load weights, then copy
    try:
        tmp_base = MelGenerator()
        tmp_wrapper = MelGeneratorTraining(tmp_base)
        # Build with same input shape
        dummy = tf.zeros([1, 64, N_MELS])
        _ = tmp_wrapper(dummy, training=False)
        tmp_wrapper.load_weights(checkpoint_path)
        _copy_matching_weights(tmp_base, model.base_model)
        logger.info("Phase 1 weights loaded via temporary model — new pose layers initialized from scratch")
        return model
    except Exception as e:
        logger.warning(f"Temporary model load failed ({e}), trying direct skip_mismatch")

    # Last resort: direct load with skip_mismatch
    try:
        model.load_weights(checkpoint_path, skip_mismatch=True)
        logger.info("Weights loaded with skip_mismatch")
    except Exception:
        logger.warning("All checkpoint loading methods failed — starting from scratch")

    return model


def _normalize_var_path(var):
    """Normalize variable path by stripping wrapper prefix and mel_generator suffix number."""
    import re
    p = var.path
    # Extract path from mel_generator onwards, normalizing mel_generator_N to mel_generator
    match = re.search(r'mel_generator(?:_\d+)?/(.*)', p)
    if match:
        return match.group(1)
    return p


def _copy_matching_weights(source_model, target_model):
    """Copy weights from source to target by normalized path, skipping mismatches."""
    source_map = {_normalize_var_path(v): v for v in source_model.variables}
    matched, skipped = 0, 0
    for tv in target_model.variables:
        key = _normalize_var_path(tv)
        sv = source_map.get(key)
        if sv is not None and sv.shape == tv.shape:
            tv.assign(sv)
            matched += 1
        else:
            skipped += 1
    logger.info(f"Weight transfer: {matched} matched, {skipped} new/skipped")


def _apply_freezing(base_model, freeze_encoder, freeze_early):
    """Freeze specified layers."""
    if freeze_encoder:
        base_model.pose_encoder.trainable = False
        logger.info("Froze pose encoder")
    if freeze_early:
        base_model.tokenizer.trainable = False
        for conv in base_model.conv_layers:
            conv.trainable = False
        base_model.transformer1.trainable = False
        logger.info("Froze tokenizer, conv layers, and transformer1")


def train_pose(data_dir, cache_dir, checkpoint_dir, batch_size, epochs, lr,
               alpha_start, alpha_end, alpha_steps, cond_dropout,
               freeze_encoder, freeze_early, resume_from=None):
    """Train pose-conditioned mel generator."""
    check_batch_size(batch_size)

    print(f"Loading pose dataset from {data_dir}")
    dataset, steps_per_epoch = create_pose_dataset(data_dir, cache_dir, batch_size)

    # Validate dataset
    for x, y, pose in dataset.take(1):
        print(f"  Mel shape: {x.shape}, Pose shape: {pose.shape}")
        tf.debugging.assert_equal(tf.shape(x)[2], N_MELS)
        tf.debugging.assert_equal(tf.shape(pose)[2], POSE_FEATURE_DIM)
    print(f"✓ Pose dataset validated ({steps_per_epoch} steps/epoch)")

    total_steps = steps_per_epoch * epochs
    lr_schedule = WarmupCosineSchedule(lr, warmup_steps=5 * steps_per_epoch, total_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)

    base_model = MelGenerator()
    model = PoseConditionedTraining(
        base_model, alpha_start=alpha_start, alpha_end=alpha_end,
        alpha_steps=alpha_steps, cond_dropout_rate=cond_dropout,
    )
    model.compile(optimizer=optimizer)

    # Build model before loading weights
    for x, y, pose in dataset.take(1):
        _ = model((x, y, pose), training=False)

    if resume_from:
        model = load_checkpoint_with_autodetect(model, resume_from)
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            model.compile(optimizer=optimizer)

    _apply_freezing(base_model, freeze_encoder, freeze_early)
    model.summary(expand_nested=True)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mel_generator_pose.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_weights_only=False, save_freq='epoch'),
        tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir / "logs"),
    ]

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    print(f"Training complete. Model saved to {checkpoint_path}")


def main():
    configure_memory()

    parser = argparse.ArgumentParser(description="Pose-conditioned mel generator training")
    parser.add_argument('--data_dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    parser.add_argument('--checkpoint_dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--alpha_start', type=float, default=0.0)
    parser.add_argument('--alpha_end', type=float, default=0.3)
    parser.add_argument('--alpha_steps', type=int, default=100000)
    parser.add_argument('--cond_dropout', type=float, default=0.0)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_early', action='store_true')

    args = parser.parse_args()
    train_pose(
        args.data_dir, args.cache_dir, args.checkpoint_dir,
        args.batch_size, args.epochs, args.lr,
        args.alpha_start, args.alpha_end, args.alpha_steps,
        args.cond_dropout, args.freeze_encoder, args.freeze_early,
        args.resume,
    )


if __name__ == '__main__':
    main()
