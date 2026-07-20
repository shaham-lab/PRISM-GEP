from typing import Optional
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from pydiffmap import diffusion_map as dm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
import pickle
from gensim import corpora
from sklearn.decomposition import NMF
import networkx as nx

X = None
CORPUS = None

def load_data(path, type):
    """
    Load data from a given path. This function should be modified to load your specific dataset.
    """
    if type == "pickle":
        with open(f'{path}/filtered_corpus.pkl', 'rb') as f:
            corpus = pickle.load(f)
        dictionary = corpora.Dictionary(corpus)
    
    elif type == "json":
        # load json file
        import json
        with open(f'{path}', 'r') as f:
            corpus = json.load(f)
        
    # load dictionary
    dictionary = corpora.Dictionary(corpus)
    dictionary.id2token = {id_: token for token, id_ in dictionary.token2id.items()}

    corpus = [" ".join(doc) for doc in corpus]  # Join tokens back into strings
    return corpus, dictionary

def load_expression_level_matrix(path):
    """
    Load expression-level matrix from a given path.
    This function should be modified to load your specific dataset.
    """
    # load tsv file containing genes expression matrix
    import pandas as pd
    from scipy import sparse

    df = pd.read_csv(path, index_col=0)
    print(f"Loaded expression matrix with shape: {df.shape}")
    return sparse.csr_matrix(df.values)

def compute_p_x(corpus):
    global X
    if X is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
    DTM = X.toarray()  # convert to dense numpy array
    word_counts = DTM.sum(axis=0)  # shape: (V,)
    p_x = word_counts / word_counts.sum()  # shape: (V,)
    return p_x

def compute_bio_px():
    """
    Compute p(x) for the bio dataset.
    X: expression level matrix (genes x cells)
    """
    global X
    # Sum across cells to get total expression per gene
    t = X.toarray()
    gene_counts = t.sum(axis=1)  # shape: (V,)
    p_x = gene_counts / gene_counts.sum()  # shape: (V,)
    return p_x


def save_dtm(path):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    # load json file
    import json
    with open(path, "r") as f:
        c = json.load(f)
        
    # Join tokens back into strings for CountVectorizer
    docs = [' '.join(doc) for doc in c]

    # Create and fit vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)  # returns sparse matrix
    vocab_order = vectorizer.get_feature_names_out().tolist()
    
    # Convert to dense matrix with word labels
    dtm = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # Save the DataFrame to a CSV file
    dtm.to_csv("bbc_dtm.csv", index=False)
    # Save the vocabulary order
    with open("bbc_vocab_order.txt", "w") as f:
        for word in vocab_order:
            f.write(f"{word}\n")
    print("DTM and vocabulary order saved.")


def get_glove_matrix(dictionary, glove_path, path_save=""):
    """
    load only the words in the dictionary
    """
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            glove[word] = vector

    # Create a matrix for the words in the dictionary
    vocab_size = len(dictionary)
    dim = len(next(iter(glove.values())))  # Get the dimension from the first vector
    glove_matrix = np.zeros((vocab_size, dim))

    for i, word in enumerate(dictionary.values()):
        if word in glove:
            glove_matrix[i] = glove[word]

    if path_save != "":
        # Save the glove matrix
        np.save(f"{path_save}/glove_matrix.npy", glove_matrix)
        print(f"Saved glove matrix to {path_save}/glove_matrix.npy")
        
    return glove_matrix


def soft_predictions(predictions):
    print("Applying softmax to predictions")
    print("Mean of predictions before softmax:", predictions.mean())
    # Define epsilon
    epsilon = 1e-06
    soft_predictions = predictions + epsilon  # Add epsilon to each element
    soft_predictions /= soft_predictions.sum(axis=1, keepdims=True) # Normalize each row so that the sum is 1

    return soft_predictions


def find_best_d(matrix):
    svd = TruncatedSVD(n_components=300)  # safely higher than needed
    X = svd.fit_transform(matrix)
    explained = np.cumsum(svd.explained_variance_ratio_)
    d1 = np.searchsorted(explained, 0.90)  # keep 90% of variance
    
    return d1



def calculate_gaussian_kernel(word_vectors, adaptive_sigma=True):
    """
    Calculate Gaussian kernel from word vectors using cosine similarity.
    
    Parameters:
    -----------
    word_vectors : numpy.ndarray
        Word vectors after SVD reduction, shape (n_words, n_dimensions)
    adaptive_sigma : bool, default=True
        Whether to use median distance as the bandwidth parameter
        
    Returns:
    --------
    kernel_matrix : numpy.ndarray
        Gaussian kernel matrix with positive similarities, shape (n_words, n_words)
    """
    cosine_sim_matrix = cosine_similarity(word_vectors)
    
    # Convert similarities to distances
    # Cosine distance = 1 - cosine similarity
    cosine_dist_matrix = 1 - cosine_sim_matrix
    
    # Determine sigma (bandwidth parameter)
    if adaptive_sigma:
        # Use median of non-zero distances as sigma
        # We exclude zeros (self-similarities) to get a better estimate
        non_zero_distances = cosine_dist_matrix[cosine_dist_matrix > 0]
        if len(non_zero_distances) > 0:
            sigma = np.median(non_zero_distances)
        else:
            sigma = 1.0  # Fallback if no non-zero distances
    else:
        # Default fixed sigma
        sigma = 1.0
    
    # Using the formula: exp(-distance²/sigma²)
    kernel_matrix = np.exp(-(cosine_dist_matrix**2) / (2 * sigma**2))
    
    return kernel_matrix, sigma


# Example usage with full pipeline
def full_svd_kernel_pipeline(sppmi_matrix):
    """
    Full pipeline from SPPMI matrix to Gaussian kernel
    
    Parameters:
    -----------
    sppmi_matrix : numpy.ndarray
        SPPMI matrix of word co-occurrences
    n_components : int, default=100
        Number of SVD components to keep
        
    Returns:
    --------
    kernel_matrix : numpy.ndarray
        Gaussian kernel matrix ready for diffusion maps
    word_vectors : numpy.ndarray
        Reduced word vectors from SVD
    """
    
    word_vectors = reduce_dim_svd(sppmi_matrix)

    # Normalize word vectors (important for consistent cosine similarities)
    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    word_vectors_normalized = word_vectors / norms
    
    # Calculate Gaussian kernel
    kernel_matrix, sigma = calculate_gaussian_kernel(word_vectors_normalized)
    
    print(f"Using sigma = {sigma}")
    
    return kernel_matrix


def diffusion_embedding(matrix, dataset, k=10):
    print("Matrix's shape before diffusion embedding:", matrix.shape)
    W = csr_matrix(matrix)
    
    # Compute D^{-1}
    row_sums = W.sum(axis=1).A1  # shape: (n,)
    D_inv = diags(np.where(row_sums != 0, 1.0 / row_sums, 0))
    
    # Compute Markov transition matrix
    P = D_inv @ W
    
    # Compute top-k eigenvectors of P
    eigenvalues, eigenvectors = eigs(P, k=k+1, which='LR')  # +1 to skip trivial eigenvector
    
    # Convert to real
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Sort by eigenvalue magnitude descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Skip the trivial (first) eigenvector if it's constant and take only positive eigenvalues
    top_k_eigenvalues = eigenvalues[1:k+1]
    top_k_eigenvectors = eigenvectors[:, 1:k+1]
    
    # Scale eigenvectors by eigenvalues (diffusion coordinates)
    scaled = top_k_eigenvectors * top_k_eigenvalues
    print("Matrix's shape after diffusion embedding:", scaled.shape)
    # save the scaled matrix to be easly loaded later
    
    np.save(f"data_ablation/{dataset}/diffusion_embedding.npy", scaled)

    return scaled

def load_diffusion_embedding(dataset, k=10):
    """
    Load precomputed diffusion embedding from file.
    """
    import numpy as np
    scaled = np.load(f"data_ablation/{dataset}/diffusion_embedding.npy")
    print("Loaded diffusion embedding with shape:", scaled.shape)
    return scaled[:, :k]  # Return only the first k components
    
    

def get_cosine_similarity(matrix):
    return cosine_similarity(matrix)


def sort_matrix_according_dictionary(matrix, vocabulary, dictionary):
    """
    Sort words according to the order in the given dictionary.
    """
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    row_indices = [word_to_index[word] for word in dictionary.values()]

    return matrix[row_indices]


def compute_pmi_matrix(corpus, dataset, kind='pmi', shift=1.0, path=None, is_first=True):
    """
    Compute PMI, PPMI, or SPPMI from a corpus.

    Parameters:
    - corpus: list of strings (documents)
    - window_size: size of sliding context window
    - kind: 'pmi', 'ppmi', or 'sppmi'
    - shift: shift value for SPPMI (typically 1.0 or 5.0)

    Returns:
    - pmi_matrix: np.array of shape (V, V)
    - vocab: list of vocabulary words
    """
    from scipy.sparse import coo_matrix

    global X
    if path is not None:
        X = load_expression_level_matrix(path)
        A = X
        vocab = None
        co_occurrence = (A.T @ A)
        co_occurrence.diagonal().fill(0)
        print(f"Co-occurrence matrix shape: {co_occurrence.shape}")

        # Convert to COO format for easy access to (i, j, val)
        coo = coo_matrix(co_occurrence)
        total = coo.data.sum()
        p_wc = coo.data / total

        row_sums = np.array(co_occurrence.sum(axis=1)).flatten()
        col_sums = np.array(co_occurrence.sum(axis=0)).flatten()
        p_w = row_sums[coo.row] / total
        p_c = col_sums[coo.col] / total

        expected = p_w * p_c
        pmi_data = np.log((p_wc + 1e-10) / (expected + 1e-10))

        # PPMI or SPPMI
        pmi_data = np.maximum(pmi_data, 0)  # for PPMI   
        
        pmi = coo_matrix((pmi_data, (coo.row, coo.col)), shape=co_occurrence.shape)
        print(f"PMI matrix shape: {pmi.shape}")
        return pmi.toarray()
    else:
        # Token count
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out()
        V = len(vocab)

        # Co-occurrence matrix using sliding window
        co_occurrence = (X.T @ X).toarray()  # shape: (V, V)
        np.fill_diagonal(co_occurrence, 0)  # remove self-co-occurrence
        print(f"Co-occurrence matrix shape: {co_occurrence.shape}")

        # Convert counts to probabilities
        total = co_occurrence.sum()
        p_wc = co_occurrence / total
        p_w = p_wc.sum(axis=1, keepdims=True)
        p_c = p_wc.sum(axis=0, keepdims=True)
        expected = p_w @ p_c

        # Compute PMI
        with np.errstate(divide='ignore'):
            pmi = np.log((p_wc + 1e-10) / (expected + 1e-10))

        if kind == 'ppmi':
            pmi = np.maximum(pmi, 0)
        elif kind == 'sppmi':
            pmi = np.maximum(pmi - np.log(shift), 0)
        elif kind != 'pmi':
            raise ValueError("kind must be 'pmi', 'ppmi', or 'sppmi'")

        # save the pmi matrix and vocabulary
        if is_first:
            import os
            if not os.path.exists(f"data/{dataset}"):
                os.makedirs(f"data/{dataset}")
                
            np.save(f"data/{dataset}/pmi_matrix.npy", pmi)
            with open(f"data/{dataset}/vocab.pkl", 'wb') as f:
                pickle.dump(vocab, f)
            print(f"Saved PMI matrix and vocabulary")
        
        return pmi, vocab

def load_pmi_matrix(path, window_size=None):
    """
    Load a precomputed PMI matrix and vocabulary from disk.
    """
    with open(f"{path}/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    if window_size is not None:
        path = f"{path}/pmi_matrix_w{int(window_size)}"
    else:
        path = f"{path}/pmi_matrix"
    pmi_matrix = np.load(f"{path}.npy")

    return pmi_matrix, vocab


def compute_tf_idf(corpus):
    """
    Build a TF-IDF matrix from the corpus.

    Parameters:
    - corpus: list of strings (documents)

    Returns:
    - tfidf_matrix: np.array of shape (N, V)
    - vocab: list of vocabulary terms
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # Compute TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(X).toarray()

    return tfidf_matrix

def scale_tfidf(tfidf_matrix):
    tfidf_mean = tfidf_matrix.mean(axis=0)  # shape: (V,)
    scaled_tfidf = np.log1p(10 * tfidf_mean)  # log(1 + 10 * x)
    
    return scaled_tfidf

def compute_affinity(sppmi, tfidf_boost, lambda_boost=1.0):
    """
    Combine SPPMI and TF-IDF into a final affinity matrix.

    Returns:
        affinity: np.array of shape (V, V)
    """
    boost_matrix = (tfidf_boost[:, None] + tfidf_boost[None, :]) / 2
    affinity = sppmi + lambda_boost * boost_matrix
    
    return affinity


def reduce_dim_dm(matrix, n_components=100):
    # Apply diffusion map
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=10, alpha=0.5, epsilon='bgh')
    embedding = mydmap.fit_transform(matrix)
    return embedding


def reduce_dim_svd(matrix):
    d = find_best_d(matrix)
    svd = TruncatedSVD(n_components=d, random_state=42)
    return svd.fit_transform(matrix)


def cluster_words(vectors, n_components=10):
    gmm = GaussianMixture(n_components=n_components,
                          verbose=1,
                          random_state=42)
    gmm.fit(vectors)
    probs = gmm.predict_proba(vectors)  # p(z | word)

    return probs, gmm.weights_

def cluster_words_vbgmm(vectors, n_components=10):
    from sklearn.mixture import BayesianGaussianMixture
    gmm = BayesianGaussianMixture(n_components=n_components,
                                  max_iter=1000,
                                  n_init=10,
                                  verbose=1,
                                  random_state=42)
    gmm.fit(vectors)
    probs = gmm.predict_proba(vectors)  # p(z | word)

    return probs, gmm.weights_


def sparsify_matrix_topk(matrix, k=50):
    """
    For each row, keep only the top-k values (by absolute value), set the rest to zero.
    """
    sparse_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        if k >= len(row):
            sparse_matrix[i] = row
            continue
        topk_idx = np.argpartition(-np.abs(row), k)[:k]
        sparse_matrix[i, topk_idx] = row[topk_idx]
    return sparse_matrix

def filter_vocabulary_by_frequency(corpus, min_count=5):
    """
    Returns filtered corpus and vocabulary, keeping only words with at least min_count occurrences.
    """
    word_counter = Counter()
    filtered_corpus = []
    for doc in corpus:
        tokens = doc.lower().split()
        word_counter.update(tokens)
        filtered_corpus.append(tokens)
    vocab = [w for w, c in word_counter.items() if c >= min_count]
    return filtered_corpus, vocab, word_counter


def compute_soc_pmi_all(pmi_matrix, top_k=100):
    V = pmi_matrix.shape[0]
    
    # Step 1: Precompute top-k neighbor indices for each word
    topk_indices = np.argsort(pmi_matrix, axis=1)[:, ::-1][:, :top_k]  # shape: (V, top_k)

    # Step 2: Initialize SOC-PMI similarity matrix
    soc_pmi_matrix = np.zeros((V, V))

    for i in range(V):
        neighbors_i = set(topk_indices[i])
        pmi_i = pmi_matrix[i]
        for j in range(i+1, V):  # matrix is symmetric
            neighbors_j = set(topk_indices[j])
            common = neighbors_i & neighbors_j
            if not common:
                continue
            soc_score = sum(pmi_i[k] + pmi_matrix[j, k] for k in common)
            soc_pmi_matrix[i, j] = soc_score
            soc_pmi_matrix[j, i] = soc_score  # symmetric

    return soc_pmi_matrix

def evaluate_prior_beta(beta_matrix, vocab, sppmi_matrix, top_n=10):
    """
    Compute topic coherence, diversity, and sparsity for the beta prior.
    beta_matrix: K x V (topic-word prior)
    vocab: list of words
    sppmi_matrix: V x V SPPMI or SOC-PMI matrix
    """
    K, V = beta_matrix.shape
    # Topic coherence: average SPPMI among top-N words per topic
    coherence_scores = []
    for k in range(K):
        top_idx = np.argsort(-beta_matrix[k])[:top_n]
        score = 0
        count = 0
        for i in range(top_n):
            for j in range(i+1, top_n):
                score += sppmi_matrix[top_idx[i], top_idx[j]]
                count += 1
        coherence_scores.append(score / max(count, 1))
    avg_coherence = np.mean(coherence_scores)
    # Diversity: average number of unique words in top-N across topics
    top_words = [set(np.argsort(-beta_matrix[k])[:top_n]) for k in range(K)]
    unique_words = set.union(*top_words)
    diversity = len(unique_words) / (K * top_n)
    # Sparsity: Gini coefficient per topic
    def gini(x):
        x = np.sort(x)
        n = len(x)
        return (1 + 1/n - 2 * np.sum((n - np.arange(n)) * x) / (n * np.sum(x)))
    sparsity = np.mean([gini(beta_matrix[k]) for k in range(K)])
    return {'avg_coherence': avg_coherence, 'diversity': diversity, 'sparsity': sparsity}

def cluster_words_nmf(matrix, n_components=10, random_state=42):
    """
    Use NMF to produce a topic-word matrix (W: words x topics), then normalize to get word-topic probabilities.
    Returns: probs (words x topics), topic weights (sum per topic)
    """
    nmf = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=500)
    W = nmf.fit_transform(np.maximum(matrix, 0))  # Ensure non-negativity
    # Normalize rows to sum to 1 (word-topic probabilities)
    probs = W / (W.sum(axis=1, keepdims=True) + 1e-10)
    topic_weights = probs.sum(axis=0)
    return probs, topic_weights

def compare_priors(prior_dict, vocab, sppmi_matrix, top_n=10):
    """
    Compare multiple priors using the same evaluation metrics.
    prior_dict: dict of {name: beta_matrix (K x V)}
    Returns: dict of {name: metrics}
    """
    results = {}
    for name, beta in prior_dict.items():
        results[name] = evaluate_prior_beta(beta, vocab, sppmi_matrix, top_n=top_n)
    return results

def create_bio_prior(matrix, dataset, path, n_components=10, k=10, path_to_save="", metric='sppmi', is_bayesian=True):
    """
    Create a prior from the matrix using GMM clustering.

    Parameters:
    - matrix: input matrix (e.g., SPPMI)
    - n_components: number of components for GMM
    - path_to_save: path to save the prior
    - metric: type of metric used (e.g., 'pmi', 'ppmi', 'sppmi')
    - is_bayesian: whether to use Bayesian inference
    """
    global X
    X = load_expression_level_matrix(path).T
    reduced_matrix = diffusion_embedding(matrix, dataset, k)
    probs, p_z = cluster_words(reduced_matrix, n_components=n_components)
    print("probs mean:", probs.mean())
    if is_bayesian:
        # p_w_given_z = p_z_given_w * p_w / p_z 
        p_z_given_w = probs
        p_x = compute_bio_px()
        probs = (p_z_given_w * p_x[:, np.newaxis]) / p_z
        print("probs mean:", probs.mean())

    probs = soft_predictions(probs)  # Smooth predictions

    if path_to_save != "":
        import torch
        import os
        path_to_save = f"{path_to_save}/{n_components}"
        print(f"Saving prior to {path_to_save}/{metric}_prior.pt")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        torch.save(probs, f"{path_to_save}/{metric}_prior.pt")    
        
def create_prior(matrix, corpus, vocab, dictionary, dataset, w=None, n_components=10, k=10, path_to_save="", metric='pmi', is_bayesian=False, smooth=False, is_first=False, eval_prior=False):
    """
    Create a prior from the matrix using GMM clustering.

    Parameters:
    - save_path: path to save the prior
    - matrix: input matrix (e.g., SPPMI)
    - n_components: number of components for GMM
    - to_save: whether to save the prior
    - metric: type of metric used (e.g., 'pmi', 'ppmi', 'sppmi')
    """
    if metric != 'glove' and is_first:
        np.fill_diagonal(matrix, 0)
        de = diffusion_embedding(matrix, dataset, w=w, k=150)
        reduced_matrix = de[:, :k]  # take only k eigenvectors
    elif metric != 'glove' and not is_first:
        de = np.load(f"data/{dataset}/diffusion_embedding.npy")
        # take only k eigenvectors
        reduced_matrix = de[:, :k]
    else:
        reduced_matrix = matrix
        
    probs, p_z = cluster_words(reduced_matrix, n_components=n_components)
    
    if is_bayesian:
        p_z_given_w = probs
        p_x = compute_p_x(corpus)
        probs = (p_z_given_w * p_x[:, np.newaxis]) / p_z # p_w_given_z
    if smooth:
        probs = soft_predictions(probs) 
    
    probs = sort_matrix_according_dictionary(probs, vocab, dictionary) 

    if path_to_save != "":
        import torch
        import os
        path_to_save = f"{path_to_save}/{n_components}"
        print(f"Saving prior to {path_to_save}/{metric}_prior.pt")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        torch.save(probs, f"{path_to_save}/{metric}_prior.pt")


def compute_ppmi_across_window_sizes(
    corpus,
    window_sizes,
    kind: str = "ppmi",
    shift: float = 1.0,
    eps: float = 1e-10,
    symmetric: bool = True,
    dataset: Optional[str] = None,
    is_first: bool = False
):
    """
    Compute PMI/PPMI/SPPMI for multiple sliding-window sizes on a text corpus.

    Parameters
    ----------
    corpus : list[str] or list[list[str]]
        Documents as raw strings or pre-tokenized lists.
    window_sizes : iterable[int]
        Context window radius(s) (e.g. [2,5,10]).
    kind : {'pmi','ppmi','sppmi'}
    shift : float
        Shift for SPPMI (typical values 1.0 or 5.0).
    min_count : int
        Minimum global token frequency to keep token in vocabulary.
    lowercase, token_pattern :
        Passed to CountVectorizer when corpus items are strings.
    eps : float
        Small constant to avoid divide-by-zero / log(0).
    symmetric : bool
        Symmetrize output matrices via max(A, A.T) if True.
    dataset : str or None
        If provided and is_first True, saved outputs will be placed under data/{dataset}/
    is_first : bool
        If True and dataset given, save the computed PMI matrices and vocab.

    Returns
    -------
    results : dict
        Mapping int(window_size) -> dict with keys:
          'pmi'  : scipy.sparse.csr_matrix (float32)
          'C'    : scipy.sparse.csr_matrix (int64) co-occurrence counts
          'vocab': list[str]
    """
    from collections import Counter
    from scipy.sparse import coo_matrix
    from sklearn.feature_extraction.text import CountVectorizer
    import os

    # Normalize docs to token lists
    if len(corpus) == 0:
        return {}

    if isinstance(corpus[0], str):
        vec = CountVectorizer()
        _ = vec.fit_transform(corpus)  # only to get analyzer
        analyzer = vec.build_analyzer()
        docs_tokens = [analyzer(doc) for doc in corpus]
        vocab = vec.get_feature_names_out()
    else:
        docs_tokens = corpus

    # global token counts and vocabulary filtering
    cnt = Counter()
    for tokens in docs_tokens:
        cnt.update(tokens)
    if len(vocab) == 0:
        raise ValueError("No tokens left after min_count filtering.")
    vocab.sort()
    token2id = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # convert docs to id lists (skip rare tokens)
    docs_ids = [[token2id[t] for t in doc if t in token2id] for doc in docs_tokens]

    results = {}
    for k in window_sizes:
        rows = []
        cols = []
        data = []
        for tokens in docs_ids:
            L = len(tokens)
            for i, w in enumerate(tokens):
                left = max(0, i - k)
                right = min(L, i + k + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    c = tokens[j]
                    rows.append(w)
                    cols.append(c)
                    data.append(1)

        if len(data) == 0:
            C = coo_matrix((V, V), dtype=np.int64).tocsr()
            R = coo_matrix((V, V), dtype=np.float32).tocsr()
            results[int(k)] = {"pmi": R, "C": C, "vocab": vocab}
            continue

        C = coo_matrix((data, (rows, cols)), shape=(V, V), dtype=np.int64).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        total = float(C.sum()) + eps
        if total <= 0:
            R = coo_matrix((V, V), dtype=np.float32).tocsr()
            results[int(k)] = {"pmi": R, "C": C, "vocab": vocab}
            continue

        Ccoo = C.tocoo()
        p_wc = Ccoo.data.astype(np.float64) / total
        row_sums = np.asarray(C.sum(axis=1)).ravel().astype(np.float64)
        col_sums = np.asarray(C.sum(axis=0)).ravel().astype(np.float64)
        p_w = row_sums / total
        p_c = col_sums / total

        pw_rows = p_w[Ccoo.row]
        pc_cols = p_c[Ccoo.col]
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi_vals = np.log((p_wc + eps) / (pw_rows * pc_cols + eps))

        if kind == "ppmi":
            vals = np.maximum(pmi_vals, 0.0)
        elif kind == "sppmi":
            vals = np.maximum(pmi_vals - np.log(max(shift, 1.0)), 0.0)
        elif kind == "pmi":
            vals = pmi_vals
        else:
            raise ValueError("kind must be 'pmi','ppmi' or 'sppmi'")

        mask = ~np.isnan(vals) & (vals != 0)
        if not np.any(mask):
            R = coo_matrix((V, V), dtype=np.float32).tocsr()
        else:
            R = coo_matrix((vals[mask].astype(np.float32),
                            (Ccoo.row[mask], Ccoo.col[mask])),
                           shape=(V, V)).tocsr()
            R.eliminate_zeros()

        if symmetric:
            R = R.maximum(R.T)

        results[int(k)] = {"pmi": R, "C": C, "vocab": vocab}

        # save if requested (one file per window size)
        if is_first and dataset is not None:
            out_dir = f"data_ablation/{dataset}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            np.save(f"{out_dir}/pmi_matrix_w{int(k)}.npy", R.toarray())
            # save vocab once (overwrite is fine)
            with open(f"{out_dir}/vocab.pkl", "wb") as f:
                pickle.dump(vocab, f)
            print(f"Saved PMI matrix (window={int(k)}) and vocabulary to {out_dir}")

    return R.toarray(), vocab