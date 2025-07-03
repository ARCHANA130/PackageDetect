import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient
# Set title
st.title("🖼️ Image Upload App")

# File uploader accepts image types
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# If a file is uploaded
if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Success message
    st.success("Image uploaded and displayed successfully!")
else:
    st.info("Please upload an image file (jpg, jpeg, png).")


# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="FXWssN8i5HJT73skntCu"
)

# infer on the uploaded image
if uploaded_file is not None:
    # Reset file pointer to start
    image_bytes = uploaded_file.read()
    #uploaded_file.seek(0)
    result = CLIENT.infer(image, model_id="classification-yng3x/3")
    pred = result["predictions"][0]
    #st.write(f"Class: {pred['class']}")
    # Display the result in a nice format
    st.markdown(
        f"""
        <div style="background-color: #222831; color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #FFD369;">Prediction Result</h4>
            <p><strong>Class:</strong> {pred['class']}</p>
            <p><strong>Confidence:</strong> {pred['confidence']:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Display the result
  