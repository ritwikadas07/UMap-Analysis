import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load the default Digits dataset
def load_digits_dataset():
    digits = load_digits()
    digits_df = pd.DataFrame(digits.data)
    digits_df['label'] = digits.target
    return digits_df, digits.images

# Function to load the Fashion MNIST dataset from CSV
def load_fashion_mnist_dataset():
    fashion_mnist_df = pd.read_csv('fashion-mnist_train_reduced.csv')
    images = fashion_mnist_df.iloc[:, 1:].values.reshape(-1, 28, 28)
    fashion_mnist_df['label'] = fashion_mnist_df.iloc[:, 0]
    return fashion_mnist_df, images

# Function to load the default animal descriptions dataset
def load_animal_descriptions():
    return pd.read_csv('animal_descriptions.csv')

# Function to load the default NAICS codes dataset
def load_naics_codes():
    return pd.read_csv('naics_codes.csv')

# Function to load the default financial statements dataset
def load_financial_statements():
    return pd.read_csv('financial_statements.csv')

# Function to load pre-trained VAE model
def load_vae_model(model_path):
    return load_model(model_path)

# Streamlit App
st.title("3D Projection of Vectors")

# Option to use default dataset or upload
dataset_choice = st.selectbox("Choose a dataset", ["Default Digits", "Default Fashion MNIST", "Default Animal Descriptions", "Default NAICS Codes", "Default Financial Statements", "Upload your own TSV file"])

# Handle the default Digits dataset
if dataset_choice == "Default Digits":
    st.write("Using the default Digits dataset.")
    df, images = load_digits_dataset()
    st.write("### Contents of the Digits Dataset")
    st.write(df.head(20))

    # Display sample images
    st.write("### Sample Images from the Digits Dataset")
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {df['label'][i]}")
        axes[i].axis('off')
    st.pyplot(fig)

    # Select only numeric data for further analysis
    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

# Handle the default Fashion MNIST dataset
elif dataset_choice == "Default Fashion MNIST":
    st.write("Using the default Fashion MNIST dataset.")
    df, images = load_fashion_mnist_dataset()
    st.write("### Contents of the Fashion MNIST Dataset")
    st.write(df.head(20))

    # Display sample images
    st.write("### Sample Images from the Fashion MNIST Dataset")
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {df['label'][i]}")
        axes[i].axis('off')
    st.pyplot(fig)

    # Select only numeric data for further analysis
    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

# Handle the default Animal Descriptions dataset
elif dataset_choice == "Default Animal Descriptions":
    st.write("Using the default Animal Descriptions dataset.")
    df = load_animal_descriptions()
    st.write("### Animal Descriptions Dataset")
    st.write(df.head(20))

    # Compute TF-IDF representation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"], columns=vectorizer.get_feature_names_out())
    st.write("### TF-IDF Vectors for Each Paragraph")
    st.write(tfidf_df.head(20))

    df = tfidf_df
    labels = df.index
    features = df

# Handle the default NAICS Codes dataset
elif dataset_choice == "Default NAICS Codes":
    st.write("Using the default NAICS Codes dataset.")
    df = load_naics_codes()
    st.write("### NAICS Codes Dataset")
    st.write(df.head(20))

    labels = df['Description']

    # Compute TF-IDF representation for descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["NAICS Code"], columns=vectorizer.get_feature_names_out())
    st.write("### TF-IDF Vectors for Each NAICS Description")
    st.write(tfidf_df.head(20))

    df = tfidf_df
    features = df

# Handle the default Financial Statements dataset
elif dataset_choice == "Default Financial Statements":
    st.write("Using the default Financial Statements dataset.")
    df = load_financial_statements()
    st.write("### Financial Statements Dataset")
    st.write(df.head(20))

    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['Company']
    features = numeric_df

# Handle the uploaded TSV file
else:
    uploaded_file = st.file_uploader("Upload the TSV file", type="tsv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep='\t')
        st.write("### First 10 Lines of the Uploaded Data")
        st.write(df.head(10))

        numeric_df = df.select_dtypes(include=[np.number])

        if 'label' in df.columns:
            labels = df['label']
            features = numeric_df.drop(columns=['label'])
        elif 'Animal' in df.columns:
            labels = df.index
            features = numeric_df
        else:
            labels = df.index
            features = numeric_df

# Perform analysis only if features and labels are set
if 'features' in locals() and 'labels' in locals():
    analysis_type = st.selectbox("Select analysis type", ["UMAP", "PCA", "VAE"])

    if analysis_type == "UMAP":
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        umap_3d_results = umap_3d.fit_transform(features)

        result_df = pd.DataFrame(umap_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D UMAP Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

    elif analysis_type == "PCA":
        pca_3d = PCA(n_components=3)
        pca_3d_results = pca_3d.fit_transform(features)

        result_df = pd.DataFrame(pca_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D PCA Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

    elif analysis_type == "VAE":
        vae_model = load_vae_model('vae_model.h5')
        vae_encoder = vae_model.get_layer('encoder')
        vae_latent = vae_encoder.predict(features)

        result_df = pd.DataFrame(vae_latent, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D VAE Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

        # Optionally, display generated images in the latent space
        st.write("### Generated Images in the Latent Space")
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))
        for i in range(5):
            decoded_img = vae_model.get_layer('decoder')(vae_latent[i].reshape(1, -1)).numpy().reshape(28, 28)
            axes[i].imshow(decoded_img, cmap='gray')
            axes[i].set_title(f"Label: {labels.iloc[i]}")
            axes[i].axis('off')
        st.pyplot(fig)

    st.write("### Analysis Results DataFrame")
    st.dataframe(result_df)

    st.plotly_chart(fig)
else:
    st.write("Please upload a TSV file to visualize the UMAP or PCA projection, or select a default dataset.")
