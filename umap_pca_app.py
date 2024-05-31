import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Function to load the default Digits dataset
def load_digits_dataset():
    # Load the Digits dataset from sklearn
    digits = load_digits()
    # Convert the dataset into a DataFrame
    digits_df = pd.DataFrame(digits.data)
    # Add the target labels to the DataFrame
    digits_df['label'] = digits.target
    # Return the DataFrame and the images
    return digits_df, digits.images

# Function to load the Fashion MNIST dataset from CSV
def load_fashion_mnist_dataset():
    # Load the Fashion MNIST dataset from a CSV file
    fashion_mnist_df = pd.read_csv('fashion-mnist_train_reduced.csv')
    # Reshape the image data
    images = fashion_mnist_df.iloc[:, 1:].values.reshape(-1, 28, 28)
    # Add the target labels to the DataFrame
    fashion_mnist_df['label'] = fashion_mnist_df.iloc[:, 0]
    # Return the DataFrame and the images
    return fashion_mnist_df, images

# Function to load the default animal descriptions dataset
def load_animal_descriptions():
    # Load the animal descriptions dataset from a CSV file
    return pd.read_csv('animal_descriptions.csv')

# Function to load the default NAICS codes dataset
def load_naics_codes():
    # Load the NAICS codes dataset from a CSV file
    return pd.read_csv('naics_codes.csv')

# Function to load the default financial statements dataset
def load_financial_statements():
    # Load the financial statements dataset from a CSV file
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
    st.write(df.head(20))  # Display only the first 20 rows of the Digits dataset

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
    st.write(df.head(20))  # Display only the first 20 rows of the Fashion MNIST dataset

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
    st.write(df.head(20))  # Display only the first 20 rows of the Animal Descriptions dataset

    # Compute TF-IDF representation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["Animal"], columns=vectorizer.get_feature_names_out())
    st.write("### TF-IDF Vectors for Each Paragraph")
    st.write(tfidf_df.head(20))  # Display only the first 20 rows of the TF-IDF vectors

    # Set df to tfidf_df for further processing
    df = tfidf_df
    labels = df.index
    features = df

# Handle the default NAICS Codes dataset
elif dataset_choice == "Default NAICS Codes":
    st.write("Using the default NAICS Codes dataset.")
    df = load_naics_codes()
    st.write("### NAICS Codes Dataset")
    st.write(df.head(20))  # Display only the first 20 rows of the NAICS Codes dataset

    # Set labels to the descriptions before TF-IDF
    labels = df['Description']

    # Compute TF-IDF representation for descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Description"])

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df["NAICS Code"], columns=vectorizer.get_feature_names_out())
    st.write("### TF-IDF Vectors for Each NAICS Description")
    st.write(tfidf_df.head(20))  # Display only the first 20 rows of the TF-IDF vectors

    # Set df to tfidf_df for further processing
    df = tfidf_df
    features = df

# Handle the default Financial Statements dataset
elif dataset_choice == "Default Financial Statements":
    st.write("Using the default Financial Statements dataset.")
    df = load_financial_statements()
    st.write("### Financial Statements Dataset")
    st.write(df.head(20))  # Display only the first 20 rows of the Financial Statements dataset

    # Ensure only numerical columns are used for analysis
    numeric_df = df.select_dtypes(include=[np.number])
    labels = df['Company']  # Use company name for hover name
    features = numeric_df

# Handle the uploaded TSV file
else:
    uploaded_file = st.file_uploader("Upload the TSV file", type="tsv")
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file, sep='\t')
        st.write("### First 10 Lines of the Uploaded Data")
        st.write(df.head(10))  # Display first 10 lines of the uploaded data

        # Use only numerical data for analysis
        numeric_df = df.select_dtypes(include=[np.number])

        # Extract features and labels for uploaded dataset
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
    st.dataframe(result_df)  # Display the analysis results DataFrame

    # Display Plotly figure
    st.plotly_chart(fig)
else:
    st.write("Please upload a TSV file to visualize the UMAP or PCA projection, or select a default dataset.")