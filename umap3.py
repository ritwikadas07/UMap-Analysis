import streamlit as st
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to process the uploaded TSV file
def process_glove_file(uploaded_file):
    word_vectors = {}
    for i, line in enumerate(uploaded_file):
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
    # Read the file and process it
    glove_vectors = process_glove_file(uploaded_file)
    
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

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each word vector
        scatter = ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'], umap_df['UMAP3'], c='b', marker='o')

        # Add labels to the points
        for i, word in enumerate(umap_df['Word']):
            ax.text(umap_df['UMAP1'][i], umap_df['UMAP2'][i], umap_df['UMAP3'][i], word, size=10, zorder=1, color='k')

        # Set labels and title
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        ax.set_title('3D UMAP Projection of GloVe Vectors')

        # Display Plot
        st.pyplot(fig)
else:
    st.write("Please upload a GloVe TSV file to visualize the UMAP projection.")
