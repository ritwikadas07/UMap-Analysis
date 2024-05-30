import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA
import streamlit as st

# Function to load GloVe vectors from a file
def load_glove_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors

# Function to get embeddings for a list of words
def get_embeddings(words, glove_vectors):
    embeddings = []
    valid_words = []
    for word in words:
        if word in glove_vectors:
            embeddings.append(glove_vectors[word])
            valid_words.append(word)
        else:
            print(f"Word '{word}' not found in GloVe vectors.")
    return np.array(embeddings), valid_words

# Load the animal names
animals_df = pd.read_csv('animals.csv')
animal_names = animals_df['Animal'].tolist()

# Load GloVe vectors
glove_file_path = 'glove.6B.50d.txt'  # Update this path to where your GloVe file is located
glove_vectors = load_glove_vectors(glove_file_path)

# Get embeddings for the animal names
animal_embeddings, valid_animal_names = get_embeddings(animal_names, glove_vectors)

# Create a DataFrame from the embeddings
embedding_df = pd.DataFrame(animal_embeddings, index=valid_animal_names)

# Save the embeddings to a CSV file
embedding_df.to_csv('animal_embeddings.csv')

# Streamlit App
st.title("3D Projection of Animal Word Embeddings")

# Select the type of analysis
analysis_type = st.selectbox("Select analysis type", ["UMAP", "PCA"])

if analysis_type == "UMAP":
    # Apply UMAP to reduce dimensions to 3D
    umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    umap_3d_results = umap_3d.fit_transform(animal_embeddings)

    # Create a DataFrame for the UMAP results
    result_df = pd.DataFrame(umap_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
    result_df['Animal'] = valid_animal_names

    # Plotting with Plotly
    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', hover_name='Animal')
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    fig.update_layout(title='3D UMAP Projection of Animal Word Embeddings',
                      scene=dict(xaxis_title='Component 1',
                                 yaxis_title='Component 2',
                                 zaxis_title='Component 3'))

else:
    # Apply PCA to reduce dimensions to 3D
    pca_3d = PCA(n_components=3)
    pca_3d_results = pca_3d.fit_transform(animal_embeddings)

    # Create a DataFrame for the PCA results
    result_df = pd.DataFrame(pca_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
    result_df['Animal'] = valid_animal_names

    # Plotting with Plotly
    fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', hover_name='Animal')
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    fig.update_layout(title='3D PCA Projection of Animal Word Embeddings',
                      scene=dict(xaxis_title='Component 1',
                                 yaxis_title='Component 2',
                                 zaxis_title='Component 3'))

# Display the DataFrame for the analysis results
st.write("### Analysis Results DataFrame")
st.dataframe(result_df)

# Display Plotly figure
st.plotly_chart(fig)
