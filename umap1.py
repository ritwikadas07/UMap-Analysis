# Import necessary libraries
import pandas as pd
from gensim.models import Word2Vec
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your text data from a TSV file into a pandas DataFrame
df = pd.read_csv('words.tsv', sep='\t')

# Ensure the DataFrame is not empty
if df.empty:
    raise ValueError("The input file is empty or not properly loaded.")

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

# Plot the results using Matplotlib
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Customize the plot with colors based on categories
scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=categories_encoded, cmap='Spectral', marker='o')

# Add a legend with category names
legend1 = ax.legend(*scatter.legend_elements(), title="Categories")
ax.add_artist(legend1)

ax.set_title('3D UMAP Projection of Text Data')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')

plt.show()
