import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import layers
import tensorflow as tf

# Define the custom Sampling layer
class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.experimental.Generator.from_seed(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = self.seed_generator.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def load_vae_model():
    with custom_object_scope({'Sampling': Sampling}):
        encoder = load_model('vae_encoder.h5', compile=False)
        decoder = load_model('vae_decoder.h5', compile=False)
        latent_space = np.load('vae_latent_space.npy')
    encoder.compile(optimizer='adam')  # Manually compile the encoder
    decoder.compile(optimizer='adam')  # Manually compile the decoder
    return encoder, decoder, latent_space

vae_encoder, vae_decoder, vae_latent = load_vae_model()

def load_fashion_mnist_dataset():
    fashion_mnist_df = pd.read_csv('fashion-mnist_train_reduced.csv')
    images = fashion_mnist_df.iloc[:, 1:].values.reshape(-1, 28, 28)
    fashion_mnist_df['label'] = fashion_mnist_df.iloc[:, 0]
    return fashion_mnist_df, images

def load_animal_descriptions():
    df = pd.read_csv('animal_descriptions.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"], columns=vectorizer.get_feature_names_out())
    labels = df["Animal"]
    return tfidf_df, labels, df

def load_naics_codes():
    df = pd.read_csv('naics_codes.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["NAICS Code"], columns=vectorizer.get_feature_names_out())
    labels = df["NAICS Code"]
    return tfidf_df, labels, df

def load_financial_statements():
    df = pd.read_csv('financial_statements.csv')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Company"], columns=vectorizer.get_feature_names_out())
    labels = df["Company"]
    return tfidf_df, labels, df

def plot_latent_space(vae_decoder, n=30, figsize=15):
    st.write("### Displaying grid of sampled digits")
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae_decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    st.pyplot(plt)

def plot_label_clusters(vae_encoder, data, labels, color_map):
    st.write("### Displaying 2D Latent Space")
    z_mean, _, _ = vae_encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=color_map)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    st.pyplot(plt)

def intro():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>VectorViz: Exploring Vector Projections</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 20px; text-align: center;'>
        <b>Author:</b> Ritwika Das for MQube.<br>
        <b>Mentor and Co-Author:</b> Rajarshi Das
    </div>
    <hr style="border:1px solid #4CAF50">
    <p style='font-size: 16px; text-align: justify;'>
        Welcome to VectorViz, an interactive tool designed to help you visualize vector projections using various dimensionality reduction techniques.
        This application has evolved through several versions, each adding new features and datasets for a more comprehensive analysis experience.
    </p>
    <div style='font-size: 16px;'>
        <b>Version 1: May 17th 2024: CSV File Upload for UMAP and PCA Analysis (3D)</b><br>
        In the initial version, VectorViz allowed users to upload their CSV files and perform 3D analysis using UMAP and PCA. This laid the foundation for visualizing complex datasets in a three-dimensional space.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 2: May 20th, 2024: Addition of VAE Analysis</b><br>
        Building on the initial capabilities, the second version introduced Variational Autoencoder (VAE) analysis, providing users with an advanced tool for uncovering the latent structures within their data.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 3: May 22nd, 2024: Integration of Animal Descriptions</b><br>
        In the third version, we enriched the application by adding a dataset of 100 sentences describing various animals. This enhancement enabled users to perform UMAP and PCA analyses in 3D on a more diverse and text-based dataset.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 4: May 24th, 2024: NAICS Codes and Financial Statements</b><br>
        The fourth version expanded the dataset options by including NAICS codes and financial statements consisting of multiple sentences. Users could now apply UMAP, PCA, and VAE analyses in 3D on these new datasets, broadening the application’s utility for business and economic data.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 5: May 30th, 2024: Inclusion of MNIST Digits and Fashion Datasets</b><br>
        In the fifth version, we introduced the popular MNIST datasets, covering both digits and fashion items. These datasets allowed users to explore their data in 3D, providing a new dimension to image-based analysis.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 6: June 3rd, 2024: 2D Latent Space Visualization for Digits MNIST</b><br>
        Recognizing the importance of 2D visualization, the sixth version added the capability to display the 2D latent space of the Digits MNIST dataset, enabling users to observe and analyze data clusters more easily.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 7: June 7th, 2024: 2D and 3D Dimensionality Reduction Options</b><br>
        The latest version gives users the flexibility to choose between 2D and 3D visualizations for their dimensionality reduction analyses, whether using UMAP, PCA, or VAE. This enhancement ensures that users can tailor their analysis to their specific needs and preferences.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to App"):
        st.session_state.page = "app"


def app():
    st.sidebar.title("Dataset and Analysis Selection")
    datasets = ["Digits MNIST", "Fashion MNIST", "Animal Descriptions", "NAICS Codes", "Financial Statements", "Upload your own TSV file"]
    dataset_choice = st.sidebar.selectbox("Choose a dataset", datasets)
    color_map = st.sidebar.selectbox("Choose a color map", ["viridis", "cividis", "plasma", "inferno"], index=0)
    analysis_types = ["UMAP", "PCA", "VAE"]
    analysis_choice = st.sidebar.selectbox("Select analysis type", analysis_types)
    dimensionality = st.sidebar.selectbox("Select dimensionality", ["2D", "3D"])

    if dataset_choice == "Upload your own CSV file":
        uploaded_file = st.sidebar.file_uploader("Upload the TSV file", type="csv")
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
    else:
        submit = st.sidebar.button("Submit")

        if submit:
            if dataset_choice == "Digits MNIST":
                st.write("Using the Digits MNIST dataset.")
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                mnist_digits = np.concatenate([x_train, x_test], axis=0)
                mnist_digits = mnist_digits[:len(mnist_digits)//4]  # Sample one-fourth of the dataset
                mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
                labels = np.concatenate([y_train, y_test])[:len(mnist_digits)]  # Sample corresponding labels

                st.write("### Sample Images from the MNIST Dataset")
                fig, axes = plt.subplots(1, 5, figsize=(10, 3))
                for i in range(5):
                    axes[i].imshow(mnist_digits[i].reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"Label: {labels[i]}")
                    axes[i].axis('off')
                st.pyplot(fig)

                numeric_df = pd.DataFrame(mnist_digits.reshape((mnist_digits.shape[0], -1)))
                features = numeric_df
                labels = pd.Series(labels)

            elif dataset_choice == "Fashion MNIST":
                st.write("Using the Fashion MNIST dataset.")
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

            elif dataset_choice == "Animal Descriptions":
                st.write("Using the Animal Descriptions dataset.")
                features, labels, df = load_animal_descriptions()
                st.write("### Animal Descriptions Dataset")
                st.write(df)

            elif dataset_choice == "NAICS Codes":
                st.write("Using the NAICS Codes dataset.")
                features, labels, df = load_naics_codes()
                st.write("### NAICS Codes Dataset")
                st.write(df.head(20))

            elif dataset_choice == "Financial Statements":
                st.write("Using the Financial Statements dataset.")
                features, labels, df = load_financial_statements()
                if features is not None and labels is not None:
                    st.write("### Financial Statements Dataset")
                    st.write(df.head(20))

            if analysis_choice == "UMAP":
                umap_model = umap.UMAP(n_components=3 if dimensionality == "3D" else 2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
                umap_results = umap_model.fit_transform(features)
                if dimensionality == "3D":
                    result_df = pd.DataFrame(umap_results, columns=['Component 1', 'Component 2', 'Component 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D UMAP Projection of Vectors',
                                      scene=dict(xaxis_title='Component 1',
                                                 yaxis_title='Component 2',
                                                 zaxis_title='Component 3'))
                    st.plotly_chart(fig)
                else:
                    result_df = pd.DataFrame(umap_results, columns=['Component 1', 'Component 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Component 1', y='Component 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D UMAP Projection of Vectors',
                                      xaxis_title='Component 1',
                                      yaxis_title='Component 2')
                    st.plotly_chart(fig)

            elif analysis_choice == "PCA":
                pca_model = PCA(n_components=3 if dimensionality == "3D" else 2)
                pca_results = pca_model.fit_transform(features)
                if dimensionality == "3D":
                    result_df = pd.DataFrame(pca_results, columns=['Component 1', 'Component 2', 'Component 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D PCA Projection of Vectors',
                                      scene=dict(xaxis_title='Component 1',
                                                 yaxis_title='Component 2',
                                                 zaxis_title='Component 3'))
                    st.plotly_chart(fig)
                else:
                    result_df = pd.DataFrame(pca_results, columns=['Component 1', 'Component 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Component 1', y='Component 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D PCA Projection of Vectors',
                                      xaxis_title='Component 1',
                                      yaxis_title='Component 2')
                    st.plotly_chart(fig)

            elif analysis_choice == "VAE":
                vae_latent_space = vae_latent
                if dimensionality == "3D":
                    if vae_latent_space.shape[1] == 3:
                        vae_3d_results = vae_latent_space[:, :3]
                    else:
                        vae_2d_results = vae_latent_space[:, :2].reshape(-1, 2)
                        vae_3d_results = np.hstack((vae_2d_results, np.zeros((vae_2d_results.shape[0], 1))))
                    result_df = pd.DataFrame(vae_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D VAE Projection of Vectors',
                                      scene=dict(xaxis_title='Component 1',
                                                 yaxis_title='Component 2',
                                                 zaxis_title='Component 3'))
                    st.plotly_chart(fig)
                else:
                    vae_2d_results = vae_latent_space[:, :2]
                    result_df = pd.DataFrame(vae_2d_results, columns=['Component 1', 'Component 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Component 1', y='Component 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D VAE Projection of Vectors',
                                      xaxis_title='Component 1',
                                      yaxis_title='Component 2')
                    st.plotly_chart(fig)

                if dataset_choice == "Digits MNIST":
                    plot_latent_space(vae_decoder)
                    plot_label_clusters(vae_encoder, mnist_digits, labels, color_map)

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 'intro'

    if st.session_state['page'] == 'intro':
        intro()
    else:
        app()

