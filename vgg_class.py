import tensorflow.keras.applications as vgg16
import tensorflow.keras.preprocessing.image as image_utils
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ProductSimilarity:
    def __init__(self, model_name, layer_name, nb_closest_images):
        self.model_name = model_name
        self.layer_name = layer_name
        self.nb_closest_images = nb_closest_images
        self.feat_extractor = self._get_feature_extractor()
    
    def _get_feature_extractor(self):
        model = self.model_name(weights='imagenet')
        feat_extractor = Model(inputs=model.input, outputs=model.get_layer(self.layer_name).output)
        return feat_extractor

    def _preprocess_images(self, images):
        processed_images = preprocess_input(images.copy())
        return processed_images

    def get_features(self, images):
        processed_images = self._preprocess_images(images)
        features = self.feat_extractor.predict(processed_images)
        return features

    def get_similarity_scores(self, product_features, all_features):
        similarity_scores = cosine_similarity(product_features, all_features)
        return similarity_scores

    def retrieve_most_similar_products(self, given_img, all_imgs, all_features):
        original = image_utils.load_img(given_img)
        plt.imshow(original)
        plt.show()

        product_features = self.get_features(image_utils.img_to_array(original))
        similarity_scores = self.get_similarity_scores(product_features, all_features)
        similarity_scores_df = pd.DataFrame(similarity_scores, columns=all_imgs, index=[given_img])
        closest_imgs = similarity_scores_df[given_img].sort_values(ascending=False)[1:self.nb_closest_images+1].index
        closest_imgs_scores = similarity_scores_df[given_img].sort_values(ascending=False)[1:self.nb_closest_images+1]

        print("most similar products:")
        for i in range(0,len(closest_imgs)):
            original = image_utils.load_img(closest_imgs[i])
            plt.imshow(original)
            plt.show()
            print("similarity score:", closest_imgs_scores[i])

# Example usage
# Set up paths and model parameters
imgs_path = "/Users/rohittiwari/Desktop/vgg_image-net/data/Apparel/Boys/Images/images_with_product_ids"
imgs_model_width, imgs_model_height = 224, 224
nb_closest_images = 4

# Initialize the ProductSimilarity class
product_similarity = ProductSimilarity(vgg16.VGG16, "fc2", nb_closest_images)

# Load the images
files = [imgs_path + "/" + x for x in os.listdir(imgs_path) if "jpg" in x]
imported_images = []
for f in files:
    original = image_utils.load_img(f, target_size=(imgs_model_width, imgs_model_height))
    imported_images.append(image_utils.img_to_array(original))
images = np.vstack(imported_images)

# Get the features for all images
all_features = product_similarity.get_features(images)

# Retrieve the most similar products for a given image
product_similarity.retrieve_most_similar_products(files[54], files, all_features)

