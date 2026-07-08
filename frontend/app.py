import os
import io
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="🍎 Fruit Freshness Detection", page_icon="🍎")

st.title("🍎 Fruit Freshness Detection")
st.write("Upload a fruit image to check whether it is **Fresh** or **Rotten**.")

# FastAPI URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": ("image.jpg", img_bytes, "image/jpeg")
            }
        )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction")

            prediction = result.get("prediction", result.get("Label", "Unknown"))
            confidence = result.get("prediction_score", result.get("confidence", 0.0))

            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Confidence:** {confidence:.4f}")

            if str(prediction).lower() == "fresh":
                st.success("🍏 Fresh Fruit")
            else:
                st.error("🍎 Rotten Fruit")

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to FastAPI at {API_URL}")