import os
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

    image_bytes = uploaded_file.getvalue()
    content_type = uploaded_file.type or "application/octet-stream"

    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": (uploaded_file.name or "image", image_bytes, content_type)
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
    except OSError as e:
        st.error(f"❌ Could not process the uploaded image: {e}")