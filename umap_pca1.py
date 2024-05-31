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
