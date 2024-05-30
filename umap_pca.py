import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA

# Function to process the uploaded TSV file
def process_glove_file(uploaded_file, limit=None):
    word_vectors = {}
    for i, line in enumerate(uploaded_file):
        if limit and i >= limit:
            break
        try:
            parts = line.decode('utf-8').split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
        except ValueError as e:
            st.warning(f"Line {i+1} could not be processed: {e}")
    return word_vectors

# Streamlit App
st.title("3D Projection of Vectors")

# Upload the TSV file
uploaded_file = st.file_uploader("Upload the TSV file", type="tsv")

if uploaded_file is not None:
    # Read the file and process it, limiting to 1000 vectors for testing
    glove_vectors = process_glove_file(uploaded_file, limit=1000)

    if len(glove_vectors) == 0:
        st.error("No valid word vectors found in the uploaded file.")
    else:
        # Create a DataFrame from the word vectors
        words = list(glove_vectors.keys())
        vectors = np.array(list(glove_vectors.values()))
        glove_df = pd.DataFrame(vectors, index=words)

        # Display the head of the DataFrame
        st.write("### First 10 Lines of the Uploaded Data")
        st.write(glove_df.head(10))

        # Select the type of analysis
        analysis_type = st.selectbox("Select analysis type", ["UMAP", "PCA"])

        if analysis_type == "UMAP":
            # Apply UMAP to reduce dimensions to 3D
            umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            umap_3d_results = umap_3d.fit_transform(glove_df)

            # Create a DataFrame for the UMAP results
            result_df = pd.DataFrame(umap_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
            result_df['Word'] = words

            # Plotting with Plotly
            fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', hover_name='Word')
            fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
            fig.update_layout(title='3D UMAP Projection of Vectors',
                              scene=dict(xaxis_title='Component 1',
                                         yaxis_title='Component 2',
                                         zaxis_title='Component 3'))

        else:
            # Apply PCA to reduce dimensions to 3D
            pca_3d = PCA(n_components=3)
            pca_3d_results = pca_3d.fit_transform(glove_df)

            # Create a DataFrame for the PCA results
            result_df = pd.DataFrame(pca_3d_results, columns=['Component 1', 'Component 2', 'Component 3'])
            result_df['Word'] = words

            # Plotting with Plotly
            fig = px.scatter_3d(result_df, x='Component 1', y='Component 2', z='Component 3', hover_name='Word')
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
    st.write("Please upload a TSV file to visualize the UMAP or PCA projection.")
