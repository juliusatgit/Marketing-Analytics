import os
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import umap




def verify_cluster_labels(df, img_path, cluster_mapping=None, n_samples=4):
    """
    Visualizes a random selection of images per cluster.
    """
    # Make sure that NA values in 'cluster' are ignored and sort the clusters
    clusters = sorted(df['cluster'].dropna().unique())
    
    for cid in clusters:
        # Dynamic titel generation (with or without name)
        if cluster_mapping and cid in cluster_mapping:
            name = cluster_mapping[cid]
            title_text = f"CLUSTER {cid}: {name.upper()}"
        else:
            title_text = f"CLUSTER {cid}"
            
        # Filter pictures for specific cluster and drop rows where 'picture_0' is NA
        items = df[df['cluster'] == cid].dropna(subset=['picture_0'])
        
        if items.empty:
            print(f"No pictures found for {title_text}")
            continue
            
        # Take random sample of items (or all if less than n_samples)
        sample = items.sample(min(n_samples, len(items)), random_state=42)
        
        

        fig, axes = plt.subplots(1, n_samples, figsize=(16, 4))
        plt.suptitle(title_text, fontsize=16, fontweight='bold')
          
        for ax, (_, row) in zip(axes, sample.iterrows()):
            full_img_path = os.path.join(img_path, str(row['picture_0']))
            
            try:
                img = Image.open(full_img_path)
                ax.imshow(img)
                ax.set_title(f"{row.get('name', '')[:20]}...", fontsize=10)
                    
            except Exception:
                ax.text(0.5, 0.5, "Image Missing", ha='center', va='center')
            
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()



def apply_umap(embedding_matrix, n_components=2, random_state=42):
    """
    Reduces the dimensionality of embeddings using UMAP.
    """
    reducer = umap.UMAP(
        n_components = n_components,
        n_neighbors = 50,
        min_dist = 0.1,
        metric = "cosine",
        random_state = random_state)
    
    return reducer.fit_transform(embedding_matrix)


   


def plot_2d_projection(X_2d, labels, title="2D Projection", palette="tab10"):
    """
    Creates a clean 2D scatter plot for cluster visualization.
    Works elegantly with noise (cluster -1 or "-1", e.g. in HDBSCAN).
    
    Parameters:
    - X_2d: Numpy array mit Form (N, 2), z.B. deine UMAP-Projektion.
    - labels: Pandas Series oder Numpy array mit den Cluster-Namen oder IDs.
    - title: Titel des Plots.
    - palette: Farbpalette für Seaborn.
    """
    plt.figure(figsize=(10, 8))
    
    # Identify noise points (works for both numeric -1 and string "-1")
    noise_mask = (labels == -1) | (labels == "-1")
    
    # Plot for actual clusters (everything that is NOT noise)
    sns.scatterplot(
        x=X_2d[~noise_mask, 0],
        y=X_2d[~noise_mask, 1],
        hue=labels[~noise_mask],
        palette=palette, 
        s=40, 
        alpha=0.8,
        legend="full"
    )
    
    # Plot Noise (if using HDBSCAN or similar)
    if noise_mask.any():
        sns.scatterplot(
            x=X_2d[noise_mask, 0],
            y=X_2d[noise_mask, 1],
            color="lightgrey", 
            s=40, 
            alpha=0.8,
            label="Noise (-1)",
            zorder=0 # Moves noise in the background
        )
    
    plt.title(title)
    plt.legend(title="Cluster")
    plt.show()