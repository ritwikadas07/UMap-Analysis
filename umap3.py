import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px

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
st.title("3D UMAP Projection of GloVe Vectors")

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

        # Apply UMAP to reduce dimensions to 3D
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        umap_3d_results = umap_3d.fit_transform(glove_df)

        # Create a DataFrame for the UMAP results
        umap_df = pd.DataFrame(umap_3d_results, columns=['UMAP1', 'UMAP2', 'UMAP3'])
        umap_df['Word'] = words

        # Display DataFrame
        st.write("### UMAP Results DataFrame")
        st.dataframe(umap_df)

        # Plotting with Plotly
        fig = px.scatter_3d(umap_df, x='UMAP1', y='UMAP2', z='UMAP3', text='Word')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(title='3D UMAP Projection of GloVe Vectors',
                          scene=dict(xaxis_title='UMAP1',
                                     yaxis_title='UMAP2',
                                     zaxis_title='UMAP3'))

        # Display Plotly figure
        st.plotly_chart(fig)
else:
    st.write("Please upload a GloVe TSV file to visualize the UMAP projection.")
