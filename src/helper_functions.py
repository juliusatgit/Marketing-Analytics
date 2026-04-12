import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import umap

import sys
from pathlib import Path

# Add src/ to path (once, so imports work)
sys.path.append(str(Path().resolve().parent / "src"))

from paths import DATA_DATASETS, DATA_EMBEDDINGS


def get_master_dataframe():
    # Load data
    transactions = pd.read_csv(DATA_DATASETS / "transactions.csv", sep=";")
    outfits = pd.read_csv(DATA_DATASETS / "outfits.csv", sep=";")
    outfit_clusters = pd.read_csv(DATA_DATASETS / "outfits_clusters_labels.csv", sep=";")


    outfits = outfits.rename(columns={"id": "outfit.id"}) # Rename for consistency with transactions

    transactions["rentalPeriod.start"] = pd.to_datetime(transactions["rentalPeriod.start"])
    transactions["rentalPeriod.end"] = pd.to_datetime(transactions["rentalPeriod.end"])

    # Join outfit meta data and cluster information
    outfits_enriched = outfits.merge(
        outfit_clusters, left_on="outfit.id", right_on="outfit.id", how="left"
    )

    # Join transactions with outfit information
    master = transactions.merge(
        outfits_enriched[["outfit.id", "cluster", "cluster_name", 
                        "pricePerWeek", "pricePerMonth", "retailPrice"]],
        on="outfit.id", how="left"
    )

    # Calculate rental duration in days
    master["rentalPeriod.start"] = pd.to_datetime(master["rentalPeriod.start"])
    master["rentalPeriod.end"]   = pd.to_datetime(master["rentalPeriod.end"])
    master["duration_days"] = (master["rentalPeriod.end"] - master["rentalPeriod.start"]).dt.days

    # Cleaning: Remove rentals with non-positive duration and missing price information
    master = master[
        (master["duration_days"] > 0) &
        (
            (master["duration_days"] <= 7) & master["pricePerWeek"].notna() |
            (master["duration_days"] > 7)  & master["pricePerMonth"].notna()
        )
    ].copy()

    # Revenue calculation based on duration and pricing
    master["revenue"] = np.where(
        master["duration_days"] <= 7,
        master["pricePerWeek"],
        master["pricePerMonth"] * master["duration_days"] / 30
    )

    print(f"Master shape: {master.shape}")
    print(master.head(2))
    
    return master




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



 

def generate_cluster_names(df, cluster_col='cluster', feature_col='category_name', top_n=2):
    """
    Automatically generates descriptive names for each cluster based on 
    the most frequent items in a specific metadata column (e.g., category_name).
    """    
    cluster_naming_dict = {}
    
    # Get unique clusters, sorted
    clusters = sorted(df[cluster_col].unique())
    
    for c in clusters:
        # Filter data for the current cluster
        cluster_data = df[df[cluster_col] == c]
              
        # Get the most frequent categories in this cluster
        top_categories_raw = cluster_data[feature_col].astype(str).str.split(",").str[0].str.strip().value_counts().head(top_n).index.tolist()

        # Clean up and remove duplicates 
        # (e.g. if Top 1 and Top 2 are somehow identical after formatting)
        unique_clean_categories = []
        for cat in top_categories_raw:
            clean_cat = str(cat).strip().title()
            if clean_cat not in unique_clean_categories:
                unique_clean_categories.append(clean_cat)
        
        # Join the clean, unique categories
        cluster_name = " & ".join(unique_clean_categories)
        
        # Save to dictionary
        cluster_naming_dict[c] = str(cluster_name)
        
        print(f"Cluster {c} contains mainly: {cluster_name}")
        
    return cluster_naming_dict




# PLOT FUNCTIONS
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