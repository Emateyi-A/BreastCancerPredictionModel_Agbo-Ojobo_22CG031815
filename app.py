import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# Define the model architecture
class BreastCancerModel(nn.Module):
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = BreastCancerModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Streamlit UI
st.title("A Breast Cancer Prediction Model")
st.markdown("Please Enter your patients measurements to predict if the tumor is Benign(is passie) or Malignant(is actie)")

column_1, column_2 = st.columns(2)

with column_1:
    radius = st.number_input("the radius", min_value=0.0, value=10.0, step=0.1)

with column_2:
    texture = st.number_input("the texture", min_value=0.0, value=10.0, step=0.1)

if st.button("the prediction", use_container_width=True):
    # Normalize input
    in_data = np.array([[radius, texture]])
    in_normal = scaler.transform(in_data)
    
    # Make prediction
    in_ten = torch.FloatTensor(in_normal)
    with torch.no_grad():
        prediction = model(in_ten).item()
    
    # Display result
    st.markdown("---")
    column_1, column_2 = st.columns(2)
    
    with column_1:
        if prediction > 0.5:
            st.error("the cancer is MALIGNANT(is passie)")
            st.write(f"Confidence: {prediction*100:.2f}%")
        else:
            st.success("the cancer is BENIGN(is actie)")
            st.write(f"the Confidence: {(1-prediction)*100:.2f}%")
    
    with column_2:
        st.metric("the Prediction Score", f"{prediction:.4f}")

