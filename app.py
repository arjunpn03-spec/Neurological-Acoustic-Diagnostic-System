import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# --- 1. UI Configuration (The High-End Look) ---
st.set_page_config(page_title="Vocal AI Diagnostic", layout="centered")
st.title("🎙️ Neurological Audio Analysis AI")
st.markdown("Upload a standard .wav voice sample (e.g., patient holding an 'Ahhh' sound) to instantly generate a Mel-Spectrogram and run it through our Convolutional Neural Network.")

# --- 2. Rebuild the CNN Architecture ---
# Streamlit needs to know the shape of the brain it is loading
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# --- 3. Load the Saved Model (Cached for Speed) ---
@st.cache_resource
def load_model():
    model = CNNModel()
    # In a real scenario, you'd load the .pth file here:
    # model.load_state_dict(torch.load("parkinsons_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 4. The Interactive Dashboard ---
st.divider()

# Give the user the option to record live OR upload a file
audio_source = st.audio_input("🔴 Record Live Voice (Hold 'Ahhhh' for 5 seconds)")

if audio_source is None:
    audio_source = st.file_uploader("Or Upload Patient Audio (.wav)", type=["wav"])

if audio_source is not None:
    # A. Play the audio
    st.audio(audio_source, format='audio/wav')
    
    if st.button("Run AI Diagnostics", type="primary"):
        with st.spinner("Analyzing frequencies and generating spectrogram..."):
            
            # B. Process Audio to Spectrogram
            audio_data, sample_rate = sf.read(audio_source)
            waveform = torch.tensor(audio_data, dtype=torch.float32)
            
            # Format to (Channels, Time) for PyTorch
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) 
            else:
                waveform = waveform.t()
                
            mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
            spectrogram = mel_transform(waveform)
            spectrogram_db = T.AmplitudeToDB()(spectrogram)
            
            # C. Render the Visual Spectrogram
            st.subheader("Visual Frequency Analysis")
            fig, ax = plt.subplots(figsize=(10, 3))
            img = ax.imshow(spectrogram_db[0].numpy(), cmap='inferno', origin='lower', aspect='auto')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title("Mel-Spectrogram (Computer Vision Input)")
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Time Frames")
            st.pyplot(fig)
            
            # D. Run the Inference (Prediction)
            spectrogram_resized = torch.nn.functional.interpolate(
                spectrogram_db.unsqueeze(0), size=(64, 64), mode='bilinear'
            )
            
            with torch.no_grad():
                prediction = model(spectrogram_resized).item()
            
            # E. Display Results Beautifully
            st.divider()
            st.subheader("Diagnostic Results")
            col1, col2 = st.columns(2)
            
            probability = prediction * 100
            
            with col1:
                st.metric(label="Parkinson's Probability", value=f"{probability:.1f}%")
            with col2:
                if probability > 50:
                    st.error("⚠️ Anomaly Detected: Visual tremors found in frequency bands. Recommend neurological follow-up.")
                else:
                    st.success("✅ Clean Audio: No significant vocal tremors detected.")