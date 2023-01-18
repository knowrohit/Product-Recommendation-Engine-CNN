import pandas as pd
import requests
import streamlit as st
from vgg_image_net import *

st.set_page_config(layout="wide")
st.title(":camera: Recom engine")


# Let user upload a picture
with st.sidebar:
    st.title("Upload a picture")

    upload_type = st.radio(
        label="How to upload the picture",
        options=(("From file", "From URL", "From webcam")),
    )

    image_bytes = None

    if upload_type == "From file":
        file = st.file_uploader(
            "Upload image file", type=[".jpg"], accept_multiple_files=False
        )
        if file:
            image_bytes = file.getvalue()

    if upload_type == "From URL":
        url = st.text_input("Paste URL")
        if url:
            image_bytes = requests.get(url).content

    if upload_type == "From webcam":
        camera = st.camera_input("Take a picture!")
        if camera:
            image_bytes = camera.getvalue()

st.write("## Uploaded picture")
if image_bytes:
    st.write("Here's what you uploaded!")
    st.image(image_bytes, width=224)
else:
    st.warning(" Please upload an image first...")
    st.stop()


st.write("## Model prediction")

# model_name = st.selectbox("Choose model", SUPPORTED_MODELS.keys())
columns = st.columns(2)
for column_index, model_name in enumerate(vgg_model.keys()):
    with columns[column_index]:
        load_model, preprocess_input, decode_predictions = vgg_model[
            feat_extractor
        ].values()

        model = load_model()
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        numpy_image = processed_image(numpy_image)
        prediction = feat_extractor.predict(processed_imgs)
        prediction_df = cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
        prediction_df.columns = ["closest_imgs", "closest_imgs_scores"]
        st.dataframe(prediction_df.sort_values(by= "closest_imgs_scores"))