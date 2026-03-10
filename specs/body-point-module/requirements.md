# Requirements

This document captures the requirements clarification process through Q&A.

---

## Q1: Input Video Format

What format and specifications will the input video have?
- Resolution (e.g., 720p, 1080p, variable)?
- Frame rate (e.g., 30fps, 60fps, variable)?
- Duration constraints (short clips, long performances, real-time streaming)?
- Single person or multiple people in frame?

**A1:** Video configuration is not yet determined (resolution and frame rate will be variable). Single person in frame. Real-time streaming capability required, similar to the audio model's streaming architecture.

---

## Q2: Temporal Window and Latency

For real-time streaming, what are the latency and temporal context requirements?
- Should the module process frame-by-frame with minimal latency (e.g., <100ms)?
- Or can it buffer a small window of frames (e.g., 1-2 seconds) for better temporal modeling?
- What's the acceptable latency budget for pose embedding generation?

**A2:** Frame-by-frame processing with minimal latency.

---

## Q3: Temporal Context for Motion Features

Since you want frame-by-frame processing but also mentioned velocity features (dx, dy) in the rough idea, how should the module maintain temporal context?
- Should it maintain a sliding history buffer (e.g., last N frames) to compute velocity?
- Or use a stateful recurrent approach (like GRU/LSTM) that maintains hidden state between frames?
- Or compute velocity only when available and handle the first frame as a special case?

**A3:** Maintain a sliding history buffer (last N frames) to compute velocity.

---

## Q4: Output Embedding Dimension and Sequence Length

What should the output embedding specifications be?
- Embedding dimension (e.g., 256, 512, match the audio model's dimension)?
- Since it's frame-by-frame, should each frame produce a single embedding vector, or multiple embedding tokens per frame?
- Should the output be compatible with a specific dimension used elsewhere in your system?

**A4:** Match the audio model's dimensions. The audio model uses D_MODEL=128 (transformer hidden dimension). Each frame should produce a single embedding vector for simplicity.

---

## Q5: Handling Missing or Low-Confidence Keypoints

How should the module handle cases where pose detection fails or produces low-confidence keypoints?
- Filter out low-confidence joints and mask them in the feature vector?
- Output a special "no pose detected" embedding?
- Use the previous frame's keypoints as fallback?
- What confidence threshold is acceptable (e.g., 0.3, 0.5)?

**A5:** No reasonable fallback exists, so all confidence thresholds are acceptable. The module should pass through all detected keypoints regardless of confidence score.

---

## Q6: Pose Encoder Architecture

For encoding the normalized skeleton features into embeddings, which approach should the module use?
- Simple MLP (faster, simpler, good baseline)
- Graph Neural Network / GCN (captures skeleton structure, more complex)
- Let this be configurable?

The rough idea mentioned both options - which is preferred for the initial design?

**A6:** Simple MLP for the initial design.

---

## Q7: Temporal Modeling for Frame-by-Frame Processing

Given frame-by-frame processing with a history buffer, how should temporal information be incorporated?
- No temporal modeling (each frame encoded independently using only velocity from buffer)
- Lightweight temporal layer (e.g., 1D conv over the history buffer)
- Recurrent layer that processes the current frame with context from history
- Since the rough idea mentioned "Temporal Transformer," should this still be included despite frame-by-frame constraints?

**A7:** Lightweight temporal layer (e.g., 1D convolution over the history buffer).

---

## Q8: Skeleton Normalization Strategy

For normalizing keypoints to body-relative coordinates, what normalization approach should be used?
- Root-relative (subtract hip midpoint, scale by torso length) as suggested in rough idea
- Bounding box normalization (normalize to 0-1 based on detected person's bounding box)
- Shoulder-width normalization (scale by shoulder distance)
- Other approach?

**A8:** Shoulder-width normalization (scale by shoulder distance).

---

## Q9: Module State Management

For real-time streaming with a history buffer, how should the module manage state?
- Stateful module that maintains internal buffer (call with single frame, module updates buffer internally)
- Stateless module where caller manages buffer (call with buffer of N frames)
- Provide both interfaces?

This affects the API design and how the module integrates with the streaming pipeline.

**A9:** Stateful module that maintains internal buffer (call with single frame, module updates buffer internally).

---

## Q10: Training and Weights

Will this module need to be trained, or will it use only pretrained components?
- Pose model: pretrained (MoveNet)
- Normalization: fixed algorithm
- MLP encoder + temporal layer: Should these be trainable or use fixed/random initialization?

If trainable, what will be the training objective (e.g., trained end-to-end with the music generation model, or pre-trained separately)?

**A10:** Pose model uses pretrained MoveNet (frozen). MLP encoder and temporal layer are trainable. No separate pre-training - these components will be trained end-to-end with the music generation model.

---

## Q11: Error Handling and Edge Cases

What should happen in edge cases?
- Frame with no person detected (MoveNet returns no keypoints or all zeros)
- Shoulder distance is zero or very small (normalization would fail)
- First frame(s) when history buffer is not yet full
- Module reset (e.g., switching to a new video stream)

Should the module output zero embeddings, skip frames, or handle these gracefully in another way?

**A11:** Edge cases will be handled outside the body module. The module can assume valid input and does not need internal error handling for these scenarios.

---

## Q12: Performance and Resource Constraints

Are there specific performance requirements or constraints?
- Target inference time per frame (e.g., must process faster than real-time at 30fps = <33ms per frame)
- GPU vs CPU deployment
- Memory constraints
- Batch processing support (process multiple frames in parallel) or strictly single-frame?

**A12:** CPU deployment for now (Mac GPU doesn't play well with TensorFlow). No specific performance targets defined yet. Single-frame processing to match the streaming requirement.

---

## Q13: History Buffer Size

How many frames should the history buffer maintain for velocity computation?

**A13:** Small window (4-8 frames) for smoother velocity computation and temporal convolution.

---

## Q14: Module Output Format

Should the module output just the embedding vector, or additional metadata?
- Just embedding: [D] tensor (e.g., [128])
- Embedding + confidence: embedding vector plus overall pose confidence score
- Embedding + raw keypoints: for debugging or downstream use
- Other information needed?

**A14:** Configurable output format. Default is just the embedding vector [D]. Optional modes can include confidence scores or raw keypoints for debugging.

---

## Q15: Testing and Validation

How should the module's correctness be validated?
- Unit tests for each component (normalization, feature building, etc.)
- Integration test with sample video
- Visual debugging tools (e.g., overlay keypoints on video frames)
- Specific acceptance criteria for the module's behavior?

**A15:** Unit tests for each component (normalization, feature building, MLP encoder, temporal layer, buffer management).

---

