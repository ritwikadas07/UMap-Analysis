import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse

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

    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['label']
    features = numeric_df.drop(columns=['label'])

# Handle the default Animal Descriptions dataset
elif dataset_choice == "Default Animal Descriptions":
    st.write("Using the default Animal Descriptions dataset.")
    df = load_animal_descriptions()
    st.write("### Animal Descriptions Dataset")
    st.write(df.head(20))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

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

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

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
        input_dim = features.shape[1]
        latent_dim = 3

        inputs = Input(shape=(input_dim,))
        h = Dense(128, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        decoder_h = Dense(128, activation='relu')
        decoder_mean = Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        vae = Model(inputs, x_decoded_mean)

        reconstruction_loss = mse(inputs, x_decoded_mean) * input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        
        vae.fit(features, epochs=50, batch_size=32, verbose=0)

        encoder = Model(inputs, z_mean)
        vae_results = encoder.predict(features)

        result_df = pd.DataFrame(vae_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D VAE Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

        # Display generated digits in the latent space
        if dataset_choice == "Default Digits":
            st.write("### Generated Digits in the Latent Space")
            decoded_imgs = vae.predict(features)
            fig, axes = plt.subplots(1, 5, figsize=(10, 3))
            for i in range(5):
                axes[i].imshow(decoded_imgs[i].reshape(8, 8), cmap='gray')
                axes[i].set_title(f"Generated: {labels[i]}")
                axes[i].axis('off')
            st.pyplot(fig)

    st.write("### Analysis Results DataFrame")
    st.dataframe(result_df)

    st.plotly_chart(fig)
else:
    st.write("Please upload a TSV file to visualize the UMAP or PCA projection, or select a default dataset.")
