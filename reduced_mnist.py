import pandas as pd

# Load the Fashion MNIST dataset
fashion_mnist_df = pd.read_csv('fashion-mnist_train.csv')

# Ensure the label column is correctly identified
label_column = 'label'
py
# Number of unique labels
num_labels = fashion_mnist_df[label_column].nunique()


samples_per_label = 4000 // num_labels

sampled_df = fashion_mnist_df.groupby(label_column).apply(lambda x: x.sample(n=samples_per_label, random_state=42)).reset_index(drop=True)

# Save the reduced dataset to a new CSV file
sampled_df.to_csv('fashion-mnist_train_reduced.csv', index=False)

print("Reduced dataset saved as 'fashion-mnist_train_reduced.csv'")
print(sampled_df[label_column].value_counts())


