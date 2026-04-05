import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import xgboost as xgb

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
def load_ensemble_and_scaler():
    nn_model = OptimizedCKDModel(input_size=10, embed_dim=32, dropout_rate=0.59)
    nn_path = 'ft_transformer_ckd_sec.pth'
    if os.path.exists(nn_path):
        try:
            nn_model.load_state_dict(torch.load(nn_path, map_location='cpu', weights_only=True))
            nn_model.eval()
        except Exception as e:
            st.error(f"Error loading NN weights: {e}")
            nn_model = None
    else:
        st.error(f"'{nn_path}' not found!")
        nn_model = None

    xgb_path = 'xgboost_ckd.pkl'
    if os.path.exists(xgb_path):
        try:
            xgb_model = joblib.load(xgb_path)
        except Exception as e:
            st.error(f"Error loading XGBoost model: {e}")
            xgb_model = None
    else:
        st.error(f"'{xgb_path}' not found!")
        xgb_model = None

    scaler_path = 'ckd_scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            scaler = None
    else:
        st.error(f"'{scaler_path}' not found!")
        scaler = None

    return nn_model, xgb_model, scaler

st.set_page_config(page_title="NephroPredict AI", layout="centered")
st.title("🩺 NephroPredict AI: CKD Risk Probability Predictor")
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

nn_model, xgb_model, scaler = load_ensemble_and_scaler()

if nn_model is not None and xgb_model is not None and scaler is not None:
    scaled_input = scaler.transform(raw_patient_input)
    
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    with torch.no_grad():
        logits = nn_model(input_tensor)
        nn_prob = torch.sigmoid(logits).item()
        
    xgb_prob = xgb_model.predict_proba(scaled_input)[:, 1][0]
    
    final_prob = (nn_prob * 0.5) + (xgb_prob * 0.5)
    
    clinical_threshold = 0.30
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="СБА қаупі (CKD Risk)", value=f"{final_prob*100:.2f}%")
        st.caption(f"🧠 NN: {nn_prob*100:.1f}% | ⚡ XGB: {xgb_prob*100:.1f}%")
    
    with col_res2:
        if final_prob >= clinical_threshold:
            st.error("🚨 Жоғары қауіп анықталды")
        else:
            st.success("✅ Төмен қауіп анықталды")
    
    st.progress(float(final_prob))
else:
    st.warning("Күтіңіз... Модельдер немесе Скейлер жүктелуде. (Please ensure .pth, .pkl, and scaler files are in the directory).")
