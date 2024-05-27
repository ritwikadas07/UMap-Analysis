import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import umap
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Streamlit title and description
st.title("3D UMAP Projection of Text Data")
st.write("This application visualizes text data in 3D using UMAP and Word2Vec.")

# File uploader to upload the words.tsv file
uploaded_file = st.file_uploader("Choose a TSV file", type="tsv")

if uploaded_file is not None:
    # Load your text data from the uploaded TSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file, sep='\t')

    # Ensure the DataFrame is not empty
    if df.empty:
        st.error("The input file is empty or not properly loaded.")
    else:
        # Extract the words and categories
        words = df['Word'].values
        categories = df['Category'].values

        # Tokenize the words
        tokenized_words = [word.split() for word in words]

        # Train a Word2Vec model
        model = Word2Vec(sentences=tokenized_words, vector_size=200, window=5, min_count=1, workers=4)

        # Get the word vectors
        # For words not in the vocabulary, we use a zero vector
        vector_dim = model.vector_size
        word_vectors = np.array([model.wv[word] if word in model.wv else np.zeros(vector_dim) for word_list in tokenized_words for word in word_list])

        # Ensure that word vectors are the same length as original words
        flattened_words = [word for word_list in tokenized_words for word in word_list]
        flattened_categories = np.repeat(categories, [len(word_list) for word_list in tokenized_words])

        # Encode categories as integers for color mapping
        le = LabelEncoder()
        categories_encoded = le.fit_transform(flattened_categories)

        # Define UMAP parameters and reduce dimensions to 3D
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        X_umap_3d = umap_3d.fit_transform(word_vectors)

        # Create a DataFrame for Plotly
        umap_df = pd.DataFrame(X_umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
        umap_df['Category'] = categories_encoded
        umap_df['Word'] = flattened_words

        # Plot the results using Plotly
        fig = px.scatter_3d(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            color='Category',
            hover_data={'Word': True, 'Category': True},
            labels={'Category': 'Categories'},
            title='3D UMAP Projection of Text Data',
            template='plotly_white'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
