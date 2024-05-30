import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_digits

# Function to load the default Digits dataset
def load_digits_dataset():
    digits = load_digits()
    digits_df = pd.DataFrame(digits.data)
    digits_df['label'] = digits.target
    return digits_df

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
dataset_choice = st.selectbox("Choose a dataset", ["Default Digits", "Default Animal Descriptions", "Default NAICS Codes", "Default Financial Statements", "Upload your own TSV file"])

if dataset_choice == "Default Digits":
    st.write("Using the default Digits dataset.")
    df = load_digits_dataset()
    st.write("### Contents of the Digits Dataset")
    st.write(df)

elif dataset_choice == "Default Animal Descriptions":
    st.write("Using the default Animal Descriptions dataset.")
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

elif dataset_choice == "Default NAICS Codes":
    st.write("Using the default NAICS Codes dataset.")
    df = load_naics_codes()
    st.write("### Contents of the NAICS Codes Dataset")
    st.write(df)

elif dataset_choice == "Default Financial Statements":
    st.write("Using the default Financial Statements dataset.")
    df = load_financial_statements()
    st.write("### Contents of the Financial Statements Dataset")
    st.write(df)

else:
    # Upload the TSV file
    uploaded_file = st.file_uploader("Upload the TSV file", type="tsv")
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file, sep='\t')
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
