import joblib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(1, num_features, embed_dim))
        self.b = nn.Parameter(torch.Tensor(1, num_features, embed_dim))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        nn.init.zeros_(self.b)
    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        return x_expanded * self.W + self.b
class FTTransformerCKD(nn.Module):
    def __init__(self, num_features, embed_dim=16, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=1e-2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    def forward(self, x):
        B = x.size(0)
        x_tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_concat = torch.cat((cls_tokens, x_tokens), dim=1)
        x_out = self.transformer(x_concat)
        cls_out = x_out[:, 0, :]
        return self.head(cls_out)
@st.cache_resource
def load_model():
    model = FTTransformerCKD(num_features=10)
    if os.path.exists('ft_transformer_ckd.pth'):
        model.load_state_dict(torch.load('ft_transformer_ckd.pth', map_location='cpu', weights_only=True))
        model.eval()
        return model
    return None
@st.cache_resource
@st.cache_resource
@st.cache_resource
def load_scaler():
    scaler_path = 'ckd_scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return None
    else:
        st.error(f"'{scaler_path}' not found!")
        return None
st.set_page_config(page_title="CKD Probability Predictor", layout="centered")
st.title("🩺 NephroPredict AI: CKD Risk Probability Predictor")
st.write("Пациенттің бастапқы клиникалық зертханалық нәтижелерін төменде енгізіңіз.")
col_input1, col_input2 = st.columns(2)
with col_input1:
    fbs = st.number_input("FastingBloodSugar / Қандағы глюкоза мөлшері (mg/dL)", min_value=50.0, max_value=400.0, value=95.0, step=1.0)
    creatinine = st.number_input("SerumCreatinine / Сарысудағы креатинин (mg/dL)", min_value=0.1, max_value=15.0, value=0.9, step=0.1)
    gfr = st.number_input("GFR / Гломерулярлы сүзу жылдамдығы (mL/min/1.73m²)", min_value=0.0, max_value=150.0, value=90.0, step=1.0)
    sodium = st.number_input("SerumElectrolytesSodium / Натрий (mEq/L)", min_value=110.0, max_value=160.0, value=140.0, step=1.0)
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

if model is None:
    st.error("Model file 'ft_transformer_ckd.pth' not found in the directory!")
else:
    scaled_input = scaler.transform(raw_patient_input)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(input_tensor)
        sensitivity_factor = 2.0 
        prob = torch.sigmoid(logits * sensitivity_factor).item()
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="СБА қаупі", value=f"{prob*100:.2f}%")
    with col_res2:
        if prob >= 0.3:
            st.error("🚨 Жоғары қауіп анықталды")
        else:
            st.success("✅ Төмен қауіп анықталды")
    st.progress(prob)