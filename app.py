import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Pneumonia Detection AI",
    layout="centered"
)

st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to get an immediate AI diagnosis.")

uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image through CNN model..."):
            try:
                bytes_data = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}

                response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code==200:
                    data = response.json()
                    prediction = data["prediction"]
                    confidence = data["confidence"] * 100
                    flagged = data["flag_for_review"]

                    st.markdown("---")
                    st.subheader("Analysis Results")

                    # Display visual result cards
                    if prediction=="PNEUMONIA":
                        st.error(f"### Diagnosis: **PNEUMONIA DETECTED**")
                    else:
                        st.success(f"### Diagnosis: **NORMAL**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                    with col2:
                        st.metric(label="Requires Review", value="Yes" if flagged else "No")

                    st.write("**Probability Breakdown:**")
                    for cls, prob in data["class_probabilities"].items():
                        st.write(f"- **{cls}:** {prob * 100:.2f}%")
                        st.progress(float(prob))

                else:
                    st.error(f"Server Error {response.status_code}: Make sure FastAPI backend is running.")

            except Exception as e:
                st.error("Could not connect to FastAPI server. Ensure Uvicorn is running on port 8000.")