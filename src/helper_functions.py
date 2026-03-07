import os
import matplotlib.pyplot as plt
from PIL import Image




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