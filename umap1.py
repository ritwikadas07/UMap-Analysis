# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

# Load your text data from a TSV file into a pandas DataFrame
df = pd.read_csv('words.tsv', sep='\t')

# Extract the words and categories
words = df['Word'].values
categories = df['Category'].values

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X = vectorizer.fit_transform(words).toarray()

# Encode categories as integers for color mapping
le = LabelEncoder()
categories_encoded = le.fit_transform(categories)

# Apply UMAP to reduce dimensionality to 3D
umap_3d = umap.UMAP(n_components=3, random_state=42)
X_umap_3d = umap_3d.fit_transform(X)

# Plot the results
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
