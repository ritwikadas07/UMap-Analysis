import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import umap.umap_ as umap
import random

# Function to load pre-trained VAE encoder and latent space
def load_vae_model():
    encoder = load_model('vae_encoder.keras')
    decoder = load_model('vae_decoder.keras')
    latent_space = np.load('latent_space.npy')
    return encoder, decoder, latent_space

# Load VAE model and latent space
vae_encoder, vae_decoder, vae_latent_space = load_vae_model()

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

# Function to load the default Animal Descriptions dataset
def load_animal_descriptions():
    df = pd.read_csv('animal_descriptions.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"], columns=vectorizer.get_feature_names_out())
    labels = df["Animal"]
    return tfidf_df, labels, df

# Function to load the default NAICS codes dataset with random samples of text data
def load_naics_codes():
    df = pd.read_csv('naics_codes_with_paragraphs.csv')
    df = df.sample(n=1000)  # Randomly sample 1000 codes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["NAICS Code"], columns=vectorizer.get_feature_names_out())
    labels = df["NAICS Code"]
    return tfidf_df, labels, df

# Function to load the default Financial Statements dataset with unique companies from the same time period
def load_financial_statements():
    df = pd.read_csv('financial_statements_with_paragraphs.csv')
    df = df[df["Time Period"] == "1Q 2024"]  # Filter for the same time period
    df = df.drop_duplicates(subset=["Company"])  # Ensure each row refers to a unique company
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Company"], columns=vectorizer.get_feature_names_out())
    labels = df["Company"]
    return tfidf_df, labels, df

# Streamlit App
st.title("3D Projection of Vectors")

dataset_choice = st.selectbox("Choose a dataset", ["Default Digits", "Default Fashion MNIST", "Default Animal Descriptions", "Default NAICS Codes", "Default Financial Statements", "Upload your own TSV file"])

if dataset_choice == "Default Digits":
    st.write("Using the default Digits dataset.")
    df, images = load_digits_dataset()
    st.write("### Contents of the Digits Dataset")
    st.write(df.head(20))

    st.write("### Sample Images from the Digits Dataset")
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {df['label'][i]}")
        axes[i].axis('off')
    st.pyplot(fig)

    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

elif dataset_choice == "Default Fashion MNIST":
    st.write("Using the default Fashion MNIST dataset.")
    df, images = load_fashion_mnist_dataset()
    st.write("### Contents of the Fashion MNIST Dataset")
    st.write(df.head(20))

    st.write("### Sample Images from the Fashion MNIST Dataset")
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {df['label'][i]}")
        axes[i].axis('off')
    st.pyplot(fig)

    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

elif dataset_choice == "Default Animal Descriptions":
    st.write("Using the default Animal Descriptions dataset.")
    features, labels, df = load_animal_descriptions()
    st.write("### Animal Descriptions Dataset")
    st.write(df)

elif dataset_choice == "Default NAICS Codes":
    st.write("Using the default NAICS Codes dataset.")
    features, labels, df = load_naics_codes()
    st.write("### NAICS Codes Dataset")
    st.write(df.head(20))

elif dataset_choice == "Default Financial Statements":
    st.write("Using the default Financial Statements dataset.")
    features, labels, df = load_financial_statements()
    st.write("### Financial Statements Dataset")
    st.write(df.head(20))

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
            labels = df['Animal']
            features = numeric_df
        else:
            labels = df.index
            features = numeric_df

if 'features' in locals() and 'labels' in locals():
    analysis_type = st.selectbox("Select analysis type", ["UMAP", "PCA", "VAE"])
    colormap = st.selectbox("Choose a colormap", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])

    if analysis_type == "UMAP":
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        umap_3d_results = umap_3d.fit_transform(features)

        result_df = pd.DataFrame(umap_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=colormap)
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

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=colormap)
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D PCA Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

    elif analysis_type == "VAE":
        vae_3d_results = vae_latent_space[:, :3]

        # Ensure labels are correctly aligned
        labels = labels.reset_index(drop=True)

        result_df = pd.DataFrame(vae_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        # Normalize the data to spread out the points
        scaler = StandardScaler()
        result_df[['Component 1', 'Component 2', 'Component 3']] = scaler.fit_transform(result_df[['Component 1', 'Component 2', 'Component 3']])

        # Add jitter to spread out the data points
        jitter_strength = 0.01
        result_df['Component 1'] += np.random.normal(0, jitter_strength, size=result_df.shape[0])
        result_df['Component 2'] += np.random.normal(0, jitter_strength, size=result_df.shape[0])
        result_df['Component 3'] += np.random.normal(0, jitter_strength, size=result_df.shape[0])

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=colormap)
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D VAE Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

        # Extra plot for 2D latent space
        vae_2d_results = vae_latent_space[:, :2]
        vae_2d_df = pd.DataFrame(vae_2d_results, columns=['Dim 1', 'Dim 2'])
        vae_2d_df['Label'] = labels

        fig2 = px.scatter(vae_2d_df, x='Dim 1', y='Dim 2', color='Label', hover_name='Label', color_continuous_scale=colormap)
        fig2.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig2.update_layout(title='2D VAE Latent Space',
                           xaxis_title='Dim 1',
                           yaxis_title='Dim 2')

    st.write("### Analysis Results DataFrame")
    st.dataframe(result_df)
    st.plotly_chart(fig)
    
    if analysis_type == "VAE":
        st.plotly_chart(fig2)
else:
    st.write("Please upload a TSV file to visualize the UMAP, PCA, or VAE projection, or select a default dataset.")
