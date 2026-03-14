from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def example():
    rng = np.random.default_rng(42)
    n_docs = 300
    n_dims = 16

    # --- Generate synthetic embedding data with cluster structure ---
    centers = {
        'sports': rng.standard_normal(n_dims) * 2 + 1,
        'tech': rng.standard_normal(n_dims) * 2 - 1,
        'science': rng.standard_normal(n_dims) * 2 + 0.5,
        'politics': rng.standard_normal(n_dims) * 2 - 0.5,
    }
    categories = rng.choice(list(centers.keys()), n_docs)
    embeddings = np.array(
        [centers[cat] + rng.standard_normal(n_dims) * 0.8 for cat in categories]
    )

    df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(n_dims)])
    df['category'] = categories
    df['relevance'] = np.clip(
        0.3
        + 0.4 * (embeddings @ rng.standard_normal(n_dims)) / n_dims
        + rng.normal(0, 0.1, n_docs),
        0,
        1,
    )
    df['doc_length'] = rng.poisson(150, n_docs) + 50
    df['citation_count'] = rng.negative_binomial(3, 0.3, n_docs)

    embed_cols = [f'dim_{i}' for i in range(n_dims)]

    # --- Dimensionality reduction ---
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(df[embed_cols])
    df['pca_x'], df['pca_y'] = pca_coords[:, 0], pca_coords[:, 1]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(df[embed_cols])
    df['tsne_x'], df['tsne_y'] = tsne_coords[:, 0], tsne_coords[:, 1]

    # --- Clustering ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[embed_cols])

    # ================================================================
    # FIGURE 1: Main dashboard
    # ================================================================
    fig1 = plt.figure(figsize=(18, 12))
    fig1.suptitle(
        'Document Embedding Analysis Dashboard', fontsize=16, fontweight='bold'
    )
    gs = GridSpec(2, 3, figure=fig1, hspace=0.35, wspace=0.3)

    # 1a - PCA scatter colored by category
    ax1 = fig1.add_subplot(gs[0, 0])
    for cat in centers:
        mask = df['category'] == cat
        ax1.scatter(
            df.loc[mask, 'pca_x'], df.loc[mask, 'pca_y'], s=20, alpha=0.6, label=cat
        )
    ax1.set_title('PCA — by Category')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax1.legend(fontsize=8)

    # 1b - t-SNE scatter colored by cluster
    ax2 = fig1.add_subplot(gs[0, 1])
    scatter = ax2.scatter(
        df['tsne_x'], df['tsne_y'], c=df['cluster'], cmap='Set1', s=20, alpha=0.6
    )
    ax2.set_title('t-SNE — by KMeans Cluster')
    plt.colorbar(scatter, ax=ax2, ticks=range(4), label='Cluster')

    # 1c - Relevance distribution per category (violin)
    ax3 = fig1.add_subplot(gs[0, 2])
    cat_list = sorted(centers.keys())
    violin_data = np.array(
        [df.loc[df['category'] == c, 'relevance'].values for c in cat_list],
        dtype=object,
    )
    parts = ax3.violinplot(
        violin_data,  # type: ignore[arg-type]
        showmedians=True,
        showextrema=False,
    )
    for pc in parts['bodies']:  # type: ignore[union-attr]
        pc.set_alpha(0.7)
    ax3.set_xticks(range(1, len(cat_list) + 1))
    ax3.set_xticklabels(cat_list, fontsize=9)
    ax3.set_title('Relevance Distribution')
    ax3.set_ylabel('Relevance')

    # 1d - Citation count vs relevance (bubble plot)
    ax4 = fig1.add_subplot(gs[1, 0])
    ax4.scatter(
        df['relevance'],
        df['citation_count'],
        s=df['doc_length'] / 5,
        alpha=0.4,
        c=df['cluster'],
        cmap='Set1',
    )
    ax4.set_xlabel('Relevance')
    ax4.set_ylabel('Citation Count')
    ax4.set_title('Citations vs Relevance (size=doc length)')

    # 1e - Heatmap of mean embeddings per category
    ax5 = fig1.add_subplot(gs[1, 1])
    mean_embed = df.groupby('category')[embed_cols].mean()
    im = ax5.imshow(mean_embed.values, aspect='auto', cmap='RdBu_r')
    ax5.set_yticks(range(len(mean_embed)))
    ax5.set_yticklabels(mean_embed.index, fontsize=9)
    ax5.set_xlabel('Embedding Dimension')
    ax5.set_title('Mean Embedding per Category')
    plt.colorbar(im, ax=ax5)

    # 1f - Explained variance (PCA scree plot)
    ax6 = fig1.add_subplot(gs[1, 2])
    pca_full = PCA().fit(df[embed_cols])
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax6.bar(
        range(1, n_dims + 1),
        pca_full.explained_variance_ratio_,
        alpha=0.6,
        label='Individual',
    )
    ax6.plot(range(1, n_dims + 1), cumvar, 'ro-', markersize=4, label='Cumulative')
    ax6.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
    ax6.set_xlabel('Principal Component')
    ax6.set_ylabel('Explained Variance Ratio')
    ax6.set_title('PCA Scree Plot')
    ax6.legend(fontsize=8)

    # ================================================================
    # FIGURE 2: Distance & hierarchy analysis
    # ================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        'Pairwise Distance & Hierarchical Clustering', fontsize=14, fontweight='bold'
    )

    # 2a - Pairwise distance heatmap (sample of 40 docs)
    sample_idx = rng.choice(n_docs, 40, replace=False)
    sample_idx.sort()
    dist_matrix = squareform(pdist(df.iloc[sample_idx][embed_cols]))
    im2 = axes2[0].imshow(dist_matrix, cmap='viridis')
    axes2[0].set_title('Pairwise Distances (40 docs)')
    plt.colorbar(im2, ax=axes2[0])

    # 2b - Dendrogram
    Z = linkage(df.groupby('category')[embed_cols].mean(), method='ward')
    dendrogram(Z, labels=sorted(centers.keys()), ax=axes2[1], leaf_font_size=10)
    axes2[1].set_title('Category Dendrogram (Ward)')

    # 2c - Nearest-neighbor distance distribution
    nn = NearestNeighbors(n_neighbors=6).fit(df[embed_cols])
    dists, _ = nn.kneighbors()
    mean_nn_dist = dists[:, 1:].mean(axis=1)  # exclude self
    for cat in cat_list:
        mask = df['category'] == cat
        axes2[2].hist(mean_nn_dist[mask], bins=25, alpha=0.5, label=cat, density=True)
    axes2[2].set_title('Mean 5-NN Distance Distribution')
    axes2[2].set_xlabel('Mean Distance to 5 Nearest Neighbors')
    axes2[2].legend(fontsize=8)

    fig2.tight_layout()

    # ================================================================
    # FIGURE 3: Correlation & statistics
    # ================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle('Feature Correlations & Statistics', fontsize=14, fontweight='bold')

    # 3a - Correlation matrix of first 8 dims + relevance
    corr_cols = [*embed_cols[:8], 'relevance']
    corr = df[corr_cols].corr()
    im3 = axes3[0].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes3[0].set_xticks(range(len(corr_cols)))
    axes3[0].set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=7)
    axes3[0].set_yticks(range(len(corr_cols)))
    axes3[0].set_yticklabels(corr_cols, fontsize=7)
    axes3[0].set_title('Correlation Matrix')
    plt.colorbar(im3, ax=axes3[0])

    # 3b - Doc length histogram by category
    for cat in cat_list:
        axes3[1].hist(
            df.loc[df['category'] == cat, 'doc_length'], bins=20, alpha=0.5, label=cat
        )
    axes3[1].set_title('Document Length Distribution')
    axes3[1].set_xlabel('Doc Length')
    axes3[1].legend(fontsize=8)

    # 3c - Category counts & mean relevance (twin axes)
    cat_counts = df['category'].value_counts().reindex(cat_list)
    cat_relevance = df.groupby('category')['relevance'].mean().reindex(cat_list)
    x_pos = np.arange(len(cat_list))
    axes3[2].bar(x_pos - 0.15, cat_counts, 0.3, color='steelblue', label='Count')
    axes3[2].set_ylabel('Document Count', color='steelblue')
    ax_twin = axes3[2].twinx()
    ax_twin.bar(x_pos + 0.15, cat_relevance, 0.3, color='coral', label='Mean Relevance')
    ax_twin.set_ylabel('Mean Relevance', color='coral')
    axes3[2].set_xticks(x_pos)
    axes3[2].set_xticklabels(cat_list, fontsize=9)
    axes3[2].set_title('Category Overview')
    lines1, labels1 = axes3[2].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axes3[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig3.tight_layout()

    # --- Print summary ---
    print(f'Dataset: {n_docs} docs, {n_dims} dims, {len(centers)} categories')
    print(f'PCA: top 2 components explain {cumvar[1]:.1%} of variance')
    print(f'KMeans cluster sizes: {np.bincount(df["cluster"])}')
    print(f'Mean relevance by category:\n{df.groupby("category")["relevance"].mean()}')

    out = Path(__file__).parent / 'plots'
    out.mkdir(exist_ok=True)
    fig1.savefig(out / 'dashboard.png', dpi=150, bbox_inches='tight')
    fig2.savefig(out / 'distances.png', dpi=150, bbox_inches='tight')
    fig3.savefig(out / 'correlations.png', dpi=150, bbox_inches='tight')
    print(f'Plots saved to {out.resolve()}')

    plt.show()
