import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA

# Function to process the uploaded TSV file
def process_tsv_file(uploaded_file, limit=None):
    df = pd.read_csv(uploaded_file, sep='\t')
    if limit:
        df = df.head(limit)
    return df

# Function to load the default MNIST dataset
def load_mnist_dataset():
    return pd.read_csv('mnist_sample.csv')

# Function to load the default TF-IDF dataset
def load_tfidf_dataset():
    return pd.read_csv('animal_essays_tfidf.csv')

# Streamlit App
st.title("3D Projection of Vectors")

# Option to use default dataset or upload
dataset_choice = st.selectbox("Choose a dataset", ["Default MNIST", "Default TF-IDF", "Upload your own TSV file"])

if dataset_choice == "Default MNIST":
    st.write("Using the default MNIST dataset.")
    df = load_mnist_dataset()
    st.write("### Contents of the MNIST Dataset")
    st.write(df)

elif dataset_choice == "Default TF-IDF":
    st.write("Using the default TF-IDF dataset.")
    df = load_tfidf_dataset()
    st.write("### Contents of the TF-IDF Dataset")
    st.write(df)

else:
    # Upload the TSV file
    uploaded_file = st.file_uploader("Upload the TSV file", type="tsv")
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_tsv_file(uploaded_file, limit=1000)
        st.write("### First 10 Lines of the Uploaded Data")
        st.write(df.head(10))

if 'df' in locals():
    # Extract features and labels
    if 'label' in df.columns:
        labels = df['label']
        features = df.drop(columns=['label'])
    elif 'Word' in df.columns:
        labels = df['Word']
        features = df.drop(columns=['Word'])
    else:
        labels = df.index
        features = df

    # Select the type of analysis
    analysis_type = st.selectbox("Select analysis type", ["UMAP", "PCA"])

    if analysis_type == "UMAP":
        # Apply UMAP to reduce dimensions to 3D
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        umap_3d_results = umap_3d.fit_transform(features)

        # Create a DataFrame for the UMAP results
        result_df = pd.DataFrame(umap_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        # Plotting with Plotly
        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D UMAP Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

    else:
        # Apply PCA to reduce dimensions to 3D
        pca_3d = PCA(n_components=3)
        pca_3d_results = pca_3d.fit_transform(features)

        # Create a DataFrame for the PCA results
        result_df = pd.DataFrame(pca_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
        result_df['Label'] = labels

        # Plotting with Plotly
        fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', color='Label', hover_name='Label')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D PCA Projection of Vectors',
                          scene=dict(xaxis_title='Component 1',
                                     yaxis_title='Component 2',
                                     zaxis_title='Component 3'))

    # Display the DataFrame for the analysis results
    st.write("### Analysis Results DataFrame")
    st.dataframe(result_df)

    # Display Plotly figure
    st.plotly_chart(fig)
else:
    st.write("Please upload a TSV file to visualize the UMAP or PCA projection, or select a default dataset.")
