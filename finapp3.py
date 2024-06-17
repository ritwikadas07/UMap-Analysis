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
import gensim.downloader as api

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
    encoder.compile(optimizer='adam')
    decoder.compile(optimizer='adam')
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
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.imshow(figure, cmap="Greys_r")
    st.pyplot(plt)

def plot_label_clusters(vae_encoder, data, labels, color_map):
    st.write("### Displaying 2D Latent Space")
    z_mean, _, _ = vae_encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=color_map)
    plt.colorbar()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(plt)

def intro():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>VectorViz: Exploring Vector Projections</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 20px; text-align: center;'>
        <b>Author:</b> Ritwika Das for MQube Cognition.<br>
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
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 8: June 11th, 2024: Setting Default Homepage, Showing TFIDF Vectorization and Dataset Descriptions</b><br>
        The latest version gives users the view of the MNIST 2D dataset as default while operating the application and adds descriptions of the datasets for user's convenience. It also shows the values of vectors after TF-IDF is applied.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>Version 9: June 17th, 2024: Added Detailed Data Information, and Second Page for Explaining Techniques</b><br>
        The latest version provides detailed information about data points and features in dataset titles and adds a second page that explains PCA, UMAP, and VAE before accessing the app.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to App"):
        st.session_state.page = "explanation"

def explanation():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Understanding Dimensionality Reduction Techniques</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 18px; text-align: justify;'>
        Dimensionality reduction is a process used to reduce the number of features or dimensions in a dataset while retaining as much information as possible. Here are the techniques used in this application:
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>1. Principal Component Analysis (PCA):</b><br>
        PCA is a statistical technique that transforms the data into a set of orthogonal components called principal components. These components explain the maximum variance in the data. PCA is widely used for feature reduction, noise reduction, and visualization of high-dimensional data.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>2. Uniform Manifold Approximation and Projection (UMAP):</b><br>
        UMAP is a non-linear dimensionality reduction technique that preserves the local and global structure of the data. It is particularly useful for visualizing clusters in high-dimensional data. UMAP is faster and often more effective than other non-linear techniques like t-SNE.
    </div>
    <div style='font-size: 16px; margin-top: 10px;'>
        <b>3. Variational Autoencoder (VAE):</b><br>
        VAE is a generative model that learns to encode data into a lower-dimensional latent space and then decode it back to the original space. VAEs are used for generating new data samples and understanding the underlying structure of the data. They provide a probabilistic framework for learning latent representations.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to App"):
        st.session_state.page = "app"

def load_sentence2vec_model():
    model = api.load('glove-wiki-gigaword-100')
    return model

def get_sentence_vectors(sentences, model):
    vectors = [np.mean([model[word] for word in sentence.split() if word in model], axis=0) for sentence in sentences]
    return np.array(vectors)

sentence2vec_model = load_sentence2vec_model()

def app():
    st.sidebar.title("Dataset and Analysis Selection")
    datasets = ["Digits MNIST (Image Data)", "Fashion MNIST (Image Data)", "Animal Descriptions (Text Data)", "NAICS Codes (Text Data)", "Financial Statements (Text Data)", "Upload your own CSV file"]
    dataset_choice = st.sidebar.selectbox("Choose a dataset", datasets)
    analysis_types = ["UMAP", "PCA", "VAE"]
    analysis_choice = st.sidebar.selectbox("Select analysis type", analysis_types)
    dimensionality = st.sidebar.selectbox("Select dimensionality", ["2D", "3D"])
    color_map = st.sidebar.selectbox("Choose a color map", ["viridis", "cividis", "plasma", "inferno"], index=0)

    submit = st.sidebar.button("Submit")

    if not submit:
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>VectorViz: Exploring Vector Projections</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Please select your options from the sidebar and click submit</h3>", unsafe_allow_html=True)
    else:
        if dataset_choice == "Upload your own CSV file":
            uploaded_file = st.sidebar.file_uploader("Upload the CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
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
            if dataset_choice == "Digits MNIST (Image Data)":
                st.write("### Digits MNIST Dataset")
                st.write("The MNIST dataset consists of 70,000 grayscale images of handwritten digits, with 60,000 for training and 10,000 for testing. Each image is 28x28 pixels.")
                st.write("**Data Points/Rows:** 70,000")
                st.write("**Features/Columns:** 784")
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                mnist_digits = np.concatenate([x_train, x_test], axis=0)
                mnist_digits = mnist_digits[:len(mnist_digits)//4]
                mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
                labels = np.concatenate([y_train, y_test])[:len(mnist_digits)]

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

            elif dataset_choice == "Fashion MNIST (Image Data)":
                st.write("### Fashion MNIST Dataset")
                st.write("The Fashion MNIST dataset contains 70,000 grayscale images of 10 categories of clothing items, with 60,000 for training and 10,000 for testing. Each image is 28x28 pixels.")
                st.write("**Data Points/Rows:** 70,000")
                st.write("**Features/Columns:** 784")
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

            elif dataset_choice == "Animal Descriptions (Text Data)":
                st.write("### Animal Descriptions Dataset")
                st.write("This dataset contains 100 sentences describing various animals. Each description is vectorized using TF-IDF.")
                st.write("**Data Points/Rows:** 100")
                st.write("**Features/Columns:** Varies by description")
                features, labels, df = load_animal_descriptions()
                st.write("### Animal Descriptions Dataset")
                st.write(df)
                st.write("### TF-IDF Vectors for Animal Descriptions")
                st.write(features)

                if analysis_choice == "Sentence2Vec":
                    sentences = df['Description'].tolist()
                    vectors = get_sentence_vectors(sentences, sentence2vec_model)
                    features = pd.DataFrame(vectors, index=df['Animal'])
                    st.write("### Sentence2Vec Vectors for Animal Descriptions")
                    st.write(features)

            elif dataset_choice == "NAICS Codes (Text Data)":
                st.write("### NAICS Codes Dataset")
                st.write("This dataset contains descriptions of various North American Industry Classification System (NAICS) codes. Each description is vectorized using TF-IDF.")
                st.write("**Data Points/Rows:** 81")
                st.write("**Features/Columns:** Varies by description")
                features, labels, df = load_naics_codes()
                st.write("### NAICS Codes Dataset")
                st.write(df)
                st.write("### TF-IDF Vectors for NAICS Codes")
                st.write(features)

            elif dataset_choice == "Financial Statements (Text Data)":
                st.write("### Financial Statements Dataset")
                st.write("This dataset contains descriptions of financial statements for year 2024 Q1 from various companies. Each description is vectorized using TF-IDF.")
                st.write("**Data Points/Rows:** 65")
                st.write("**Features/Columns:** Varies by description")
                st.write("**Note:** Data for the companies was created by ChatGPT")
                features, labels, df = load_financial_statements()
                st.write("### Financial Statements Dataset")
                st.write(df)
                st.write("### TF-IDF Vectors for Financial Statements")
                st.write(features)

            if analysis_choice == "UMAP":
                umap_model = umap.UMAP(n_components=3 if dimensionality == "3D" else 2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
                umap_results = umap_model.fit_transform(features)
                if dimensionality == "3D":
                    result_df = pd.DataFrame(umap_results, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D UMAP Projection of Vectors',
                                      scene=dict(xaxis_title='Dimension 1',
                                                 yaxis_title='Dimension 2',
                                                 zaxis_title='Dimension 3'))
                    st.plotly_chart(fig)
                else:
                    result_df = pd.DataFrame(umap_results, columns=['Dimension 1', 'Dimension 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Dimension 1', y='Dimension 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D UMAP Projection of Vectors',
                                      xaxis_title='Dimension 1',
                                      yaxis_title='Dimension 2')
                    st.plotly_chart(fig)

            elif analysis_choice == "PCA":
                pca_model = PCA(n_components=3 if dimensionality == "3D" else 2)
                pca_results = pca_model.fit_transform(features)
                if dimensionality == "3D":
                    result_df = pd.DataFrame(pca_results, columns=['Eigen Vector 1', 'Eigen Vector 2', 'Eigen Vector 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Eigen Vector 1', y='Eigen Vector 2', z='Eigen Vector 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D PCA Projection of Vectors',
                                      scene=dict(xaxis_title='Eigen Vector 1',
                                                 yaxis_title='Eigen Vector 2',
                                                 zaxis_title='Eigen Vector 3'))
                    st.plotly_chart(fig)
                else:
                    result_df = pd.DataFrame(pca_results, columns=['Eigen Vector 1', 'Eigen Vector 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Eigen Vector 1', y='Eigen Vector 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D PCA Projection of Vectors',
                                      xaxis_title='Eigen Vector 1',
                                      yaxis_title='Eigen Vector 2')
                    st.plotly_chart(fig)

            elif analysis_choice == "VAE":
                vae_latent_space = vae_latent
                if dimensionality == "3D":
                    if vae_latent_space.shape[1] == 3:
                        vae_3d_results = vae_latent_space[:, :3]
                    else:
                        vae_2d_results = vae_latent_space[:, :2].reshape(-1, 2)
                        vae_3d_results = np.hstack((vae_2d_results, np.zeros((vae_2d_results.shape[0], 1))))
                    result_df = pd.DataFrame(vae_3d_results, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter_3d(result_df, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='3D VAE Projection of Vectors',
                                      scene=dict(xaxis_title='Dimension 1',
                                                 yaxis_title='Dimension 2',
                                                 zaxis_title='Dimension 3'))
                    st.plotly_chart(fig)
                else:
                    vae_2d_results = vae_latent_space[:, :2]
                    result_df = pd.DataFrame(vae_2d_results, columns=['Dimension 1', 'Dimension 2'])
                    result_df['Label'] = labels.astype(str)
                    fig = px.scatter(result_df, x='Dimension 1', y='Dimension 2', color='Label', hover_name='Label', color_continuous_scale=color_map)
                    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
                    fig.update_layout(title='2D VAE Projection of Vectors',
                                      xaxis_title='Dimension 1',
                                      yaxis_title='Dimension 2')
                    st.plotly_chart(fig)

                if dataset_choice == "Digits MNIST (Image Data)":
                    plot_latent_space(vae_decoder)
                    plot_label_clusters(vae_encoder, mnist_digits, labels, color_map)

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 'intro'

    if st.session_state['page'] == 'intro':
        intro()
    elif st.session_state['page'] == 'explanation':
        explanation()
    else:
        app()
