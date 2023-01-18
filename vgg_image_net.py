from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

imgs_path = "/Users/rohittiwari/Desktop/vgg_image-net/data/Apparel/Boys/Images/images_with_product_ids"
imgs_model_width, imgs_model_height = 224, 224
nb_closest_images = 4

vgg_model = vgg16.VGG16(weights='imagenet')
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
feat_extractor.summary()

files = [imgs_path + "/" + x for x in os.listdir(imgs_path) if "jpg" in x]
original = load_img(files[0], target_size=(imgs_model_width, imgs_model_height))

numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)
processed_image = preprocess_input(image_batch.copy())
img_features = feat_extractor.predict(processed_image)

importedImages = []

for f in files:
    filename = f
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    importedImages.append(image_batch)
    
images = np.vstack(importedImages)
processed_imgs = preprocess_input(images.copy())

img_features = feat_extractor.predict(processed_imgs)
cosSimilarities = cosine_similarity(img_features)
cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)

def retrieve_most_similar_products(given_img):
    print("original product:")

    original = load_img(given_img, target_size=(imgs_model_width, imgs_model_height))
    plt.imshow(original)
    plt.show()

    print("most similar products:")
    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1]

    for i in range(0,len(closest_imgs)):
        original = load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height))
        plt.imshow(original)
        plt.show()
        print("similarity score : ",closest_imgs_scores[i])

retrieve_most_similar_products(files[54])