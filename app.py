import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class OptimizedCKDModel(nn.Module):
    def __init__(self, input_size, embed_dim=32, dropout_rate=0.59):
        super(OptimizedCKDModel, self).__init__()
        
        self.input_size = input_size
        self.embed_dim = embed_dim
        
        self.W = nn.Parameter(torch.Tensor(1, input_size, embed_dim))
        self.b = nn.Parameter(torch.Tensor(1, input_size, embed_dim))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        nn.init.zeros_(self.b)
        
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True, dropout=dropout_rate/2)
        self.norm = nn.LayerNorm(self.embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        tokens = x_expanded * self.W + self.b
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        out = self.classifier(tokens)
        return out

@st.cache_resource
def load_model():
    model = OptimizedCKDModel(input_size=10, embed_dim=32, dropout_rate=0.59)
    model_path = 'ft_transformer_ckd.pth'
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Critical Error loading weights: {e}")
            return None
    return None

@st.cache_resource
def load_scaler():
    scaler_path = 'ckd_scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return None
    return None

st.set_page_config(page_title="NephroPredict AI", layout="centered")
st.title("🩺 NephroPredict AI")
st.write("Пациенттің клиникалық зертханалық нәтижелерін енгізіңіз.")

col1, col2 = st.columns(2)

with col1:
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", value=95.0)
    creatinine = st.number_input("Serum Creatinine (mg/dL)", value=0.9)
    gfr = st.number_input("GFR (mL/min/1.73m²)", value=90.0)
    sodium = st.number_input("Sodium (mEq/L)", value=140.0)
    chol_total = st.number_input("Total Cholesterol (mg/dL)", value=180.0)

with col2:
    hba1c = st.number_input("HbA1c (%)", value=5.5)
    bun = st.number_input("BUN Levels (mg/dL)", value=15.0)
    protein = st.number_input("Protein In Urine (0-4+)", value=0.0)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", value=14.0)
    chol_hdl = st.number_input("HDL Cholesterol (mg/dL)", value=50.0)

st.divider()
model = load_model()
scaler = load_scaler()

if model and scaler:
    raw_input = [[fbs, hba1c, creatinine, bun, gfr, protein, sodium, hemoglobin, chol_total, chol_hdl]]
    scaled_input = scaler.transform(raw_input)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits * 2.0).item()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("СБА қаупі", f"{prob*100:.2f}%")
    with c2:
        if prob >= 0.3:
            st.error("🚨 Жоғары қауіп")
        else:
            st.success("✅ Төмен қауіп")
    st.progress(prob)
