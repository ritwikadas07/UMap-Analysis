import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to process the uploaded TSV file
def process_tsv_file(uploaded_file, limit=None):
    df = pd.read_csv(uploaded_file, sep='\t')
    if limit:
        df = df.head(limit)
    return df

# Function to load the default Digits dataset
def load_digits_dataset():
    return pd.read_csv('digits_sample.csv')

# Function to load the default animal descriptions dataset
def load_animal_descriptions():
    return pd.read_csv('animal_descriptions.csv')

# Streamlit App
st.title("3D Projection of Vectors")

# Option to use default dataset or upload
dataset_choice = st.selectbox("Choose a dataset", ["Default Digits", "Default Animal Descriptions", "Upload your own TSV file"])

if dataset_choice == "Default Digits":
    st.write("Using the default Digits dataset.")
    df = load_digits_dataset()
    st.write("### Contents of the Digits Dataset")
    st.write(df)

elif dataset_choice == "Default Animal Descriptions":
    st.write("Using the default animal descriptions dataset.")
    df = load_animal_descriptions()
    st.write("### First 100 Sentences of the Animal Descriptions Dataset")
    st.write(df.head(100))

    # Compute TF-IDF representation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"].head(100))

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"].head(100), columns=vectorizer.get_feature_names_out())
    st.write("### TF-IDF Vectors for Each Paragraph")
    st.write(tfidf_df)

    # Set df to tfidf_df for further processing
    df = tfidf_df

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
    elif 'Animal' in df.columns:
        labels = df.index
        features = df
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
