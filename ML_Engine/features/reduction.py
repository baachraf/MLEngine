"""
Dimensionality Reduction
========================

Functions for dimensionality reduction techniques.
Note: PCA is also available in feature_selection.py as apply_pca().
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ..data.transformation import encode_categorical

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False
    warnings.warn('umap-learn not installed. UMAP dimensionality reduction will not be available.')


def apply_pca(X, n_components=None, variance=0.95, should_encode_categorical=True):
    """
    Apply Principal Component Analysis (PCA).
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    n_components : int, optional
        Number of components. If None, use variance threshold.
    variance : float, default=0.95
        Variance to retain (used if n_components is None)
    should_encode_categorical : bool, default=True
        Whether to encode categorical columns using label encoding
    
    Returns
    -------
    X_pca : array-like
        Transformed data
    pca : PCA
        Fitted PCA object
    """
    # Handle categorical encoding if needed
    if should_encode_categorical and isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X, _ = encode_categorical(X, cat_cols, method='label')
    
    pca = PCA(n_components=n_components)
    
    if n_components is None:
        # Fit PCA to determine number of components for variance threshold
        pca.fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance) + 1
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def apply_tsne(X, n_components=2, perplexity=30.0, random_state=42, should_encode_categorical=True):
    """
    Apply t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    n_components : int, default=2
        Number of dimensions for embedding (2 or 3)
    perplexity : float, default=30.0
        t-SNE perplexity parameter
    random_state : int, default=42
        Random seed
    should_encode_categorical : bool, default=True
        Whether to encode categorical columns using label encoding
    
    Returns
    -------
    X_tsne : array-like
        Embedded data
    tsne : TSNE
        Fitted t-SNE object
    """
    # Handle categorical encoding if needed
    if should_encode_categorical and isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X, _ = encode_categorical(X, cat_cols, method="label")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X)
    return X_tsne, tsne


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, should_encode_categorical=True):
    """
    Apply Uniform Manifold Approximation and Projection (UMAP).
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    n_components : int, default=2
        Number of dimensions for embedding
    n_neighbors : int, default=15
        Number of neighbors
    min_dist : float, default=0.1
        Minimum distance between points in embedding
    random_state : int, default=42
        Random seed
    
    Returns
    -------
    X_umap : array-like
        Embedded data
    reducer : umap.UMAP
        Fitted UMAP object
    
    Raises
    ------
    ImportError
        If umap-learn package is not installed
    """
    if not UMAP_AVAILABLE:
        raise ImportError('umap-learn is required for apply_umap')
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer


def get_available_methods():
    """
    Get list of available dimensionality reduction methods.
    
    Returns
    -------
    list
        Method names
    """
    methods = ['pca', 'tsne']
    if UMAP_AVAILABLE:
        methods.append('umap')
    return methods


def reduce_dimensions(X, method='pca', **kwargs):
    """
    Unified interface for dimensionality reduction.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    method : {'pca', 'tsne', 'umap'}
        Dimensionality reduction method
    **kwargs
        Additional arguments passed to the specific method
    
    Returns
    -------
    X_reduced : array-like
        Reduced data
    model
        Fitted reduction model
    """
    if method == 'pca':
        return apply_pca(X, **kwargs)
    elif method == 'tsne':
        return apply_tsne(X, **kwargs)
    elif method == 'umap':
        return apply_umap(X, **kwargs)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def apply_dimensionality_reduction(X, method='pca', **kwargs):
    """
    Alias for reduce_dimensions for backward compatibility.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    method : {'pca', 'tsne', 'umap'}
        Dimensionality reduction method
    **kwargs
        Additional arguments passed to the specific method
    
    Returns
    -------
    X_reduced : array-like
        Reduced data
    model
        Fitted reduction model
    """
    return reduce_dimensions(X, method=method, **kwargs)

