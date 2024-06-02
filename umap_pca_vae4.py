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
    fashion_mnist_df = pd.read_csv('/mnt/data/fashion-mnist_train_reduced.csv')
    images = fashion_mnist_df.iloc[:, 1:].values.reshape(-1, 28, 28)
    fashion_mnist_df['label'] = fashion_mnist_df.iloc[:, 0]
    return fashion_mnist_df, images

# Function to load the default Animal Descriptions dataset
def load_animal_descriptions():
    df = pd.read_csv('/mnt/data/animal_descriptions.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"], columns=vectorizer.get_feature_names_out())
    labels = df["Animal"]
    return tfidf_df, labels, df

# Function to load the default NAICS codes dataset with random samples of text data
def load_naics_codes():
    df = pd.read_csv('/mnt/data/naics_codes_sampled.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["NAICS Code"], columns=vectorizer.get_feature_names_out())
    labels = df["NAICS Code"]
    return tfidf_df, labels, df

# Function to load the default Financial Statements dataset with unique companies from the same time period
def load_financial_statements():
    df = pd.read_csv('/mnt/data/financial_statements_filtered.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Company"], columns=vectorizer.get_feature_names_out())
    labels = df["Company"]
    return tfidf_df, labels, df

# Function to visualize images
def visualize_images(images, labels, title):
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.suptitle(title)
    st.pyplot(fig)

# Function to perform analysis and plot results
def analyze_and_plot(features, labels, analysis_type, title, colormap):
    if analysis_type == "UMAP":
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        results = umap_3d.fit_transform(features)
    elif analysis_type == "PCA":
        pca_3d = PCA(n_components=3)
        results = pca_3d.fit_transform(features)
    elif analysis_type == "VAE":
        results = vae_latent_space[:, :3]
        scaler = StandardScaler()
        results = scaler.fit_transform(results)
        jitter_strength = 0.01
        results += np.random.normal(0, jitter_strength, size=results.shape)
    else:
        raise ValueError("Invalid analysis type")

    result_df = pd.DataFrame(results, columns=['Component 1', 'Component 2', 'Component 3'])
    result_df['Label'] = labels

    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=colormap)
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    fig.update_layout(title=title,
                      scene=dict(xaxis_title='Component 1',
                                 yaxis_title='Component 2',
                                 zaxis_title='Component 3'))
    st.plotly_chart(fig)
    
    return result_df

# Function to reconstruct images from latent space and visualize them
def reconstruct_and_visualize(latent_space, decoder, title):
    reconstructed_images = decoder.predict(latent_space)
    # Ensure the reshaping matches the correct dimensions
    if reconstructed_images.shape[-1] == 784:  # For Digits and Fashion MNIST datasets (28x28)
        reconstructed_images = reconstructed_images.reshape(-1, 28, 28)
    else:
        raise ValueError("Unexpected shape for reconstructed images")

    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        axes[i].imshow(reconstructed_images[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    st.pyplot(fig)

# Function to generate a lattice of points in the latent space and decode them
def generate_lattice_and_decode(decoder, grid_size=20):
    # Create a grid of points in the latent space
    grid_x = np.linspace(-2, 2, grid_size)
    grid_y = np.linspace(-2, 2, grid_size)
    grid = np.array([[x, y] for x in grid_x for y in grid_y])

    # Decode the grid points
    decoded_images = decoder.predict(grid).reshape(grid_size, grid_size, 28, 28)

    # Visualize the decoded images in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(decoded_images[i, j], cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle('Lattice of Decoded Images from VAE Latent Space')
    st.pyplot(fig)

# Streamlit App
st.title("3D Projection of Vectors")

dataset_choice = st.selectbox("Choose a dataset", ["Default Digits", "Default Fashion MNIST", "Default Animal Descriptions", "Default NAICS Codes", "Default Financial Statements", "Upload your own TSV file"])

if dataset_choice == "Default Digits":
    st.write("Using the default Digits dataset.")
    df, images = load_digits_dataset()
    st.write("### Contents of the Digits Dataset")
    st.write(df.head(20))
    st.write("### Sample Images from the Digits Dataset")
    visualize_images(images, df['label'], "Sample Images from the Digits Dataset")
    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

elif dataset_choice == "Default Fashion MNIST":
    st.write("Using the default Fashion MNIST dataset.")
    df, images = load_fashion_mnist_dataset()
    st.write("### Contents of the Fashion MNIST Dataset")
    st.write(df.head(20))
    st.write("### Sample Images from the Fashion MNIST Dataset")
    visualize_images(images, df['label'], "Sample Images from the Fashion MNIST Dataset")
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

    result_df = analyze_and_plot(features, labels, analysis_type, f"3D {analysis_type} Projection of Vectors", colormap)
    
    if analysis_type == "VAE" and dataset_choice in ["Default Digits", "Default Fashion MNIST"]:
        vae_2d_results = vae_latent_space[:, :2]
        vae_2d_df = pd.DataFrame(vae_2d_results, columns=['Dim 1', 'Dim 2'])
        vae_2d_df['Label'] = labels
        fig2 = px.scatter(vae_2d_df, x='Dim 1', y='Dim 2', color='Label', hover_name='Label', color_continuous_scale=colormap)
        fig2.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig2.update_layout(title='2D VAE Latent Space',
                           xaxis_title='Dim 1',
                           yaxis_title='Dim 2')
        st.plotly_chart(fig2)

        # Reconstruct images from the VAE latent space
        st.write("### Reconstructed Images from VAE Latent Space")
        reconstruct_and_visualize(vae_latent_space[:5], vae_decoder, "Reconstructed Images from VAE Latent Space")

        # Generate and display lattice of decoded images
        st.write("### Lattice of Decoded Images from VAE Latent Space")
        generate_lattice_and_decode(vae_decoder)

else:
    st.write("Please upload a TSV file to visualize the UMAP, PCA, or VAE projection, or select a default dataset.")
