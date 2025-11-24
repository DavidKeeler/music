import tensorflow as tf
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from complex_layers import ComplexDense, ComplexTransformerBlock
from pose_detection import ComplexPoseEncoder, MultimodalFusion, preprocess_video
from complex_music_model.audio_transformer_freq import AudioTransformerFreq
from complex_music_model.train_audio import complex_stft_loss, load_audio_data

# Constants
FRAME_LENGTH = 1024
FRAME_STEP = 512
SAMPLE_RATE = 22050

class MultimodalAudioModel(tf.keras.Model):
    """Audio model with pose conditioning."""
    
    def __init__(self, audio_model, pose_encoder, fusion_layer, **kwargs):
        super().__init__(**kwargs)
        self.audio_model = audio_model
        self.pose_encoder = pose_encoder
        self.fusion_layer = fusion_layer
        
    def call(self, inputs, training=False):
        audio_input, pose_input = inputs
        
        # Get pose embeddings
        pose_embeddings = self.pose_encoder(pose_input, training=training)
        
        # Get audio features from first layers
        x = self.audio_model.input_dense(audio_input)
        x = self.audio_model.input_dense2(x)
        
        # Fuse with pose
        x = self.fusion_layer(x, pose_embeddings, training=training)
        
        # Continue through audio model
        x = self.audio_model.conv1d(x, training=training)
        x = self.audio_model.conv1d_transpose(x, training=training)
        x = self.audio_model.transformer_block(x, training=training)
        x = self.audio_model.output_dense(x)
        x = self.audio_model.output_dense2(x)
        
        # STFT consistency
        return self.audio_model.stft_layer(x)

class MultimodalTrainer(tf.keras.Model):
    """Training wrapper for multimodal model."""
    
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        
    def train_step(self, data):
        (audio_x, pose_x), y = data
        
        with tf.GradientTape() as tape:
            pred = self.base_model([audio_x, pose_x], training=True)
            total_loss, mag_loss, phase_loss, continuity_loss, smooth_loss = complex_stft_loss(y, pred)
        
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else g for g in grads]
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            'loss': total_loss,
            'mag_loss': mag_loss,
            'phase_loss': phase_loss,
            'continuity_loss': continuity_loss,
            'smooth_loss': smooth_loss
        }

def create_multimodal_dataset(audio_data, video_paths, seq_len=32, batch_size=4):
    """Create dataset with audio and pose data."""
    
    def data_generator():
        for i in range(len(video_paths)):
            # Load pose data
            pose_data = preprocess_video(video_paths[i])
            if pose_data is None:
                continue
                
            # Get corresponding audio segment
            audio_segment = audio_data[i:i+seq_len]
            pose_segment = pose_data[:seq_len]
            
            # Pad if needed
            if len(pose_segment) < seq_len:
                padding = seq_len - len(pose_segment)
                pose_segment = tf.pad(pose_segment, [[0, padding], [0, 0], [0, 0]])
            
            # Create input/target pairs
            audio_input = audio_segment[:-1]
            audio_target = audio_segment[1:]
            pose_input = pose_segment[:-1]
            
            yield (audio_input, pose_input), audio_target
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            (tf.TensorSpec([seq_len-1, 513], tf.complex64),
             tf.TensorSpec([seq_len-1, 14, 2], tf.float32)),
            tf.TensorSpec([seq_len-1, 513], tf.complex64)
        )
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_multimodal():
    """Main training function."""
    print(f"=== Multimodal Training Started: {datetime.now()} ===")
    
    # Load audio data
    audio_data = load_audio_data()
    print(f"Audio data shape: {audio_data.shape}")
    
    # Video paths (you'll need to provide these)
    video_paths = [
        # Add your video file paths here
        # "/path/to/video1.mp4",
        # "/path/to/video2.mp4",
    ]
    
    if not video_paths:
        print("No video paths provided. Add video files to train multimodal model.")
        return
    
    # Create models
    audio_model = AudioTransformerFreq()
    pose_encoder = ComplexPoseEncoder(embedding_dim=256)
    fusion_layer = MultimodalFusion(audio_dim=256, pose_dim=256, fusion_type='cross_attention')
    
    # Load pretrained audio weights
    checkpoint_path = "/Users/davidkeeler/models/conducting/checkpoints/autoreg_weights.weights.h5"
    if os.path.exists(checkpoint_path):
        audio_model.load_weights(checkpoint_path)
        print(f"Loaded audio weights from: {checkpoint_path}")
    
    # Create multimodal model
    multimodal_model = MultimodalAudioModel(audio_model, pose_encoder, fusion_layer)
    trainer = MultimodalTrainer(multimodal_model)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Create dataset
    dataset = create_multimodal_dataset(audio_data, video_paths, seq_len=32, batch_size=2)
    
    # Training parameters
    epochs = 100
    
    # Checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "/Users/davidkeeler/models/conducting/checkpoints/multimodal_weights.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor='loss'
    )
    
    # Train
    print(f"--- Training | seq_len=32 | epochs={epochs} ---")
    history = trainer.fit(
        dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint_callback]
    )
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_multimodal()
