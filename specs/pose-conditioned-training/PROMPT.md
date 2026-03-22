# PROMPT.md — Pose-Conditioned Training

## Objective

Integrate pose (body point) conditioning into the MelGenerator training pipeline. Add cross-attention to TransformerBlock, update MelGenerator with optional pose input, create a new `train_pose.py` script for phased pose-conditioned training, and implement alignment losses.

## Key Requirements

- Add `build_temporal_pose_encoder` to `src/body_point_module/encoder.py`: `[B, T, 85]` → `[B, T, 128]`, variable-length, TimeDistributed MLP + causal Conv1D
- Add `enable_cross_attn=False` flag to `TransformerBlock` in `src/music_generation/layers.py` with optional `cross_attn_kv` and `cross_attn_alpha` in `call()`. When disabled or kv=None, behavior is identical to current.
- Update `MelGenerator` in `src/music_generation/model.py`: add `pose=None` to `call()`, `forward_tokens()`, `forward_from_tokens()`, `generate()`. Pose path: temporal encoder → Dense(D_MODEL) → Conv1D(stride=4) → cross-attention in transformer2/transformer3. transformer1 stays unconditional. `pose_alpha` variable gates contribution.
- Add `create_pose_dataset` to `src/music_generation/dataset.py`: yields `(mel, mel, pose)` triples from `.wav` + `{stem}_pose.npy` pairs. Skips files without pose data.
- Create `src/music_generation/train_pose.py` with `PoseConditionedTraining` wrapper and CLI. Supports: cosine alpha scheduling (`--alpha_start`, `--alpha_end`, `--alpha_steps`), per-sample conditioning dropout (`--cond_dropout`), layer freezing (`--freeze_encoder`, `--freeze_early`), auto-detect checkpoint type on `--resume`.
- Add `PoseAudioAlignmentLoss` and `OnsetAlignmentLoss` to `src/music_generation/losses.py`. Implemented but NOT wired into training.
- Add `POSE_FEATURE_DIM = 85`, `POSE_EMBEDDING_DIM = 128` to `src/music_generation/config.py`
- Existing `train.py` and all existing tests must pass without modification.

## Acceptance Criteria

- Given MelGenerator with no pose input, when `call(mel)` is invoked, then output is identical to current model
- Given MelGenerator with pose and `pose_alpha=0.0`, when forward pass runs, then output equals no-pose output
- Given MelGenerator with pose and `pose_alpha=1.0`, when forward pass runs, then output differs from no-pose output
- Given pose features `[B, T, 85]`, when passed through temporal encoder, then output shape is `[B, T, 128]`
- Given `--freeze_encoder`, when training runs, then pose encoder weights do not change
- Given Phase 1 checkpoint, when `train_pose.py` loads it, then new layers auto-initialize and training starts
- Given `--alpha_start 0.0 --alpha_end 0.3 --alpha_steps 100000`, when 100k steps complete, then alpha ≈ 0.3
- Given `--cond_dropout 0.15` and batch of 16, then ~2-3 samples have pose masked per step
- Given audio files with matching `_pose.npy`, when pose dataset created, then yields `(mel, mel, pose)` with correct shapes
- Given audio file without pose file, when dataset iterates, then file is skipped with warning

## Reference

Full spec at `specs/pose-conditioned-training/` — see `design.md` for architecture details and `plan.md` for implementation steps.
