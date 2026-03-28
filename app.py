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
    # These parameters must be 1:1 with your final training script
    model = OptimizedCKDModel(input_size=10, embed_dim=32, dropout_rate=0.59)
    model_path = 'ft_transformer_ckd.pth'
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            return None
    return None

@st.cache_resource
def load_scaler():
    scaler_path = 'ckd_scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            # Using joblib as identified in the file structure
            return joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return None
    else:
        st.error(f"'{scaler_path}' not found!")
        return None

st.set_page_config(page_title="NephroPredict AI", layout="centered")
st.title("命️ NephroPredict AI: CKD Risk Probability Predictor")
st.write("Пациенттің бастапқы клиникалық зертханалық нәтижелерін төменде енгізіңіз.")

col_input1, col_input2 = st.columns(2)

with col_input1:
    fbs = st.number_input("FastingBloodSugar / Қандағы глюкоза (mg/dL)", min_value=50.0, max_value=400.0, value=95.0, step=1.0)
    creatinine = st.number_input("SerumCreatinine / Сарысудағы креатинин (mg/dL)", min_value=0.1, max_value=15.0, value=0.9, step=0.1)
    gfr = st.number_input("GFR / Гломерулярлы сүзу жылдамдығы (mL/min/1.73m²)", min_value=0.0, max_value=150.0, value=90.0, step=1.0)
    sodium = st.number_input("Sodium / Натрий (mEq/L)", min_value=110.0, max_value=160.0, value=140.0, step=1.0)
    chol_total = st.number_input("CholesterolTotal / Жалпы холестерин (mg/dL)", min_value=100.0, max_value=400.0, value=180.0, step=1.0)

with col_input2:
    hba1c = st.number_input("HbA1c / Гликирленген гемоглобин (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)
    bun = st.number_input("BUNLevels / Қандағы азот мочевинасы (mg/dL)", min_value=5.0, max_value=100.0, value=15.0, step=1.0)
    protein = st.number_input("ProteinInUrine / Зәрдегі белок деңгейі (0-4+)", min_value=0.0, max_value=4.0, value=0.0, step=0.5)
    hemoglobin = st.number_input("HemoglobinLevels / Гемоглобин деңгейі (g/dL)", min_value=5.0, max_value=20.0, value=14.0, step=0.1)
    chol_hdl = st.number_input("CholesterolHDL / HDL холестерині (mg/dL)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)

raw_patient_input = [[fbs, hba1c, creatinine, bun, gfr, protein, sodium, hemoglobin, chol_total, chol_hdl]]

st.divider()

model = load_model()
scaler = load_scaler()

if model is not None and scaler is not None:
    # 1. Scaling raw inputs to Z-scores
    scaled_input = scaler.transform(raw_patient_input)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    # 2. Prediction pass
    with torch.no_grad():
        logits = model(input_tensor)
        # Apply sensitivity factor for clinical safety margin
        sensitivity_factor = 2.0 
        prob = torch.sigmoid(logits * sensitivity_factor).item()
    
    # 3. Results Visualization
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="СБА қаупі (CKD Risk)", value=f"{prob*100:.2f}%")
    
    with col_res2:
        if prob >= 0.3:
            st.error("🚨 Жоғары қауіп анықталды")
        else:
            st.success("✅ Төмен қауіп анықталды")
    
    st.progress(prob)
else:
    st.warning("Күтіңіз... Модель немесе Скейлер жүктелуде.")