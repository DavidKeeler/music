"""Vocoder loss components for HiFi-GAN fine-tuning."""

import tensorflow as tf


class MultiResolutionSTFTLoss(tf.keras.layers.Layer):
    """Multi-resolution STFT loss with multiple FFT sizes."""
    
    def __init__(self, fft_sizes, hop_sizes, win_sizes, **kwargs):
        super().__init__(**kwargs)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        
    def call(self, pred_audio, target_audio):
        """Compute L1 loss on magnitude spectrograms at multiple resolutions."""
        total_loss = 0.0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            pred_spec = tf.signal.stft(
                pred_audio,
                frame_length=win_size,
                frame_step=hop_size,
                fft_length=fft_size,
                window_fn=tf.signal.hann_window
            )
            target_spec = tf.signal.stft(
                target_audio,
                frame_length=win_size,
                frame_step=hop_size,
                fft_length=fft_size,
                window_fn=tf.signal.hann_window
            )
            
            # Magnitude
            pred_mag = tf.abs(pred_spec)
            target_mag = tf.abs(target_spec)
            
            # L1 loss
            loss = tf.reduce_mean(tf.abs(pred_mag - target_mag))
            total_loss += loss
            
        return total_loss / len(self.fft_sizes)


class SpectralConvergenceLoss(tf.keras.layers.Layer):
    """Spectral convergence loss: ||S_pred - S_true||_F / ||S_true||_F"""
    
    def call(self, pred_spec, target_spec):
        """Compute spectral convergence loss."""
        numerator = tf.norm(pred_spec - target_spec, ord='fro')
        denominator = tf.norm(target_spec, ord='fro')
        return numerator / (denominator + 1e-8)


class FeatureMatchingLoss(tf.keras.layers.Layer):
    """Feature matching loss on discriminator intermediate features."""
    
    def call(self, pred_features, target_features):
        """Compute L1 loss on discriminator features."""
        total_loss = 0.0
        
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss = tf.reduce_mean(tf.abs(pred_feat - target_feat))
            total_loss += loss
            
        return total_loss / len(pred_features)


class AdversarialLoss(tf.keras.layers.Layer):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type='hinge', **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        
    def generator_loss(self, fake_scores):
        """Generator loss: maximize discriminator output on fake samples."""
        if self.loss_type == 'hinge':
            return -tf.reduce_mean(fake_scores)
        elif self.loss_type == 'lsgan':
            return tf.reduce_mean(tf.square(fake_scores - 1.0))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def discriminator_loss(self, real_scores, fake_scores):
        """Discriminator loss: maximize real, minimize fake."""
        if self.loss_type == 'hinge':
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_scores))
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_scores))
            return real_loss + fake_loss
        elif self.loss_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real_scores - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_scores))
            return real_loss + fake_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class SubBandSTFTLoss(tf.keras.layers.Layer):
    """Sub-band STFT loss: split into frequency bands and compute STFT loss per band."""
    
    def __init__(self, num_bands=3, fft_size=1024, hop_size=256, win_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.num_bands = num_bands
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        
    def call(self, pred_audio, target_audio):
        """Compute STFT loss on frequency sub-bands."""
        # Compute full STFT
        pred_spec = tf.signal.stft(
            pred_audio,
            frame_length=self.win_size,
            frame_step=self.hop_size,
            fft_length=self.fft_size,
            window_fn=tf.signal.hann_window
        )
        target_spec = tf.signal.stft(
            target_audio,
            frame_length=self.win_size,
            frame_step=self.hop_size,
            fft_length=self.fft_size,
            window_fn=tf.signal.hann_window
        )
        
        # Split into bands
        num_freqs = tf.shape(pred_spec)[1]
        band_size = num_freqs // self.num_bands
        
        total_loss = 0.0
        for i in range(self.num_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < self.num_bands - 1 else num_freqs
            
            pred_band = tf.abs(pred_spec[:, start_idx:end_idx, :])
            target_band = tf.abs(target_spec[:, start_idx:end_idx, :])
            
            loss = tf.reduce_mean(tf.abs(pred_band - target_band))
            total_loss += loss
            
        return total_loss / self.num_bands
