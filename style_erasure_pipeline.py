import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from concept_erasure import LeaceEraser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def is_top_k_match(embeddings_0, embeddings_1, k=3):
    
    # Combine both lists into a single array
    all_embeddings = np.concatenate((embeddings_0, embeddings_1), axis=0)  # shape (512, embedding_size)
    num_cases = len(embeddings_0)
    top_k_matches = []

    for i in range(num_cases):
        case_embedding = embeddings_0[i].reshape(1, -1)  # Reshape for the cosine_similarity function

        # Calculate cosine similarity between this case embedding and all embeddings
        similarities = cosine_similarity(case_embedding, all_embeddings).flatten()

        # Get the similarity score with the corresponding oral argument
        corresponding_similarity = similarities[num_cases + i]  # index of the corresponding oral argument

        # Find the top k similarity scores (excluding the case embedding itself)
        top_k_indices = np.argsort(similarities)[-(k+1):-1]  # Get the indices of the top 3 similarities
        top_k_similarities = similarities[top_k_indices]  # Corresponding similarity scores

        # Check if the corresponding similarity is in the top k
        is_top_k = corresponding_similarity >= np.min(top_k_similarities)
        top_k_matches.append(is_top_k)

    return top_k_matches

def get_exact_pairs(X, k, text_list_1):

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Determine cluster labels for source_1 and source_2 embeddings
    labels_1 = clusters[:len(text_list_1)]
    labels_2 = clusters[len(text_list_1):]

    # Count exact matches (source_1, source_2 pairs) in each cluster
    matches_per_cluster = Counter()

    for i, label_1 in enumerate(labels_1):
        # Check if the corresponding embedding_2 has the same cluster label
        if label_1 == labels_2[i]:
            matches_per_cluster[label_1] += 1

    # Print the number of matches per cluster
    total_pairs = 0
    for cluster_id in range(k):
        total_pairs += matches_per_cluster[cluster_id]

    return total_pairs

def load_embedding_model(embedding, trust_remote_code=True) -> SentenceTransformer:
    model_map = {
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'jina': 'jinaai/jina-embeddings-v3',
        'gist': 'avsolatorio/GIST-small-Embedding-v0',
        'nv': 'nvidia/NV-Embed-v2',
        'mini': 'sentence-transformers/all-MiniLM-L6-v2', 
        'e5': 'intfloat/multilingual-e5-large-instruct',
    }

    if embedding not in model_map:
        raise ValueError(f"Unknown embedding type: '{embedding}'")

    model_name = model_map[embedding]
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    return model

def run_erasure_two_sources(text_list_1, text_list_2, embedding='mini', k=5, top_k_retrieval=20, max_n_clusters=32): 
    
    # Load the sentence embedding model
    model = load_embedding_model(embedding)

    # Combine texts and label them
    text_list = text_list_1 + text_list_2
    labels = ['source_1'] * len(text_list_1) + ['source_2'] * len(text_list_2)

    # Compute embeddings
    embeddings = model.encode(text_list)
    embeddings = torch.from_numpy(embeddings)

    # Convert string labels to numeric
    numeric_labels = torch.tensor([0 if label == 'source_1' else 1 for label in labels])

    # Run LEACE erasure
    eraser = LeaceEraser.fit(embeddings, numeric_labels)
    embeddings_erased = eraser(embeddings)

    ### Run k-means clustering - before erasure
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings.numpy())

    # Count the density distribution for each source in each cluster
    cluster_counts = {cluster: [0, 0] for cluster in range(k)}
    for label, source in zip(kmeans_labels, numeric_labels):
        cluster_counts[label][source.item()] += 1

    source_1_counts = [cluster_counts[i][0] for i in range(k)]
    source_2_counts = [cluster_counts[i][1] for i in range(k)]

    # Plotting 
    sns.set_style("whitegrid")
    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bar1 = ax.bar(x - width/2, source_1_counts, width, label='Source 1', color='midnightblue')
    bar2 = ax.bar(x + width/2, source_2_counts, width, label='Source 2', color='darkorange')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Density Count')
    ax.set_title('Cluster Density Distribution by Source Before Erasure', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(k)], rotation=45)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig('kmeans_before_erasure.png')
    plt.close()

    ### Run k-means clustering - after erasure
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings_erased.numpy())

    # Count the density distribution for each source in each cluster
    cluster_counts = {cluster: [0, 0] for cluster in range(k)}
    for label, source in zip(kmeans_labels, numeric_labels):
        cluster_counts[label][source.item()] += 1

    source_1_counts = [cluster_counts[i][0] for i in range(k)]
    source_2_counts = [cluster_counts[i][1] for i in range(k)]

    # Plotting
    x = np.arange(k)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bar1 = ax.bar(x - width/2, source_1_counts, width, label='Source 1', color='midnightblue')
    bar2 = ax.bar(x + width/2, source_2_counts, width, label='Source 2', color='darkorange')

    ax.set_xlabel('Clusters')
    ax.set_ylabel('Density Count')
    ax.set_title('Cluster Density Distribution by Source After Erasure', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(k)], rotation=45)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig('kmeans_after_erasure.png')
    plt.close()

    ### Perform PCA - before erasure
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Separate PCA results by source
    source_1_pca = pca_result[:len(text_list_1)]
    source_2_pca = pca_result[len(text_list_1):]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(source_1_pca[:, 0], source_1_pca[:, 1], color='midnightblue', label='Source 1', alpha=0.7, s=15)  # Smaller points
    plt.scatter(source_2_pca[:, 0], source_2_pca[:, 1], color='darkorange', label='Source 2', alpha=0.7, s=15)  # Smaller points
    plt.title('PCA of Embeddings Before Erasure', fontsize=20)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pca_before_erasure.png')
    plt.close()

    ### Perform PCA - after erasure
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_erased)

    # Separate PCA results by source
    source_1_pca = pca_result[:len(text_list_1)]
    source_2_pca = pca_result[len(text_list_1):]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(source_1_pca[:, 0], source_1_pca[:, 1], color='midnightblue', label='Source 1', alpha=0.7, s=15)  # Smaller points
    plt.scatter(source_2_pca[:, 0], source_2_pca[:, 1], color='darkorange', label='Source 2', alpha=0.7, s=15)  # Smaller points
    plt.title('PCA of Embeddings After Erasure', fontsize=20)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('pca_after_erasure.png')
    plt.close()

    ### Generate top k rankings
    
    # Split back into two sets 
    embeddings_1 = embeddings[:len(text_list_1)]
    embeddings_2 = embeddings[len(text_list_1):]
    embeddings_erased_1 = embeddings_erased[:len(text_list_1)]
    embeddings_erased_2 = embeddings_erased[len(text_list_1):]
    
    ret_list, ret_list_erased = [], []
    for tk in range(top_k_retrieval, 0, -1):

        top_k_results = is_top_k_match(embeddings_1, embeddings_2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list.append(prc)

        top_k_results = is_top_k_match(embeddings_erased_1, embeddings_erased_2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list_erased.append(prc)

    # X-axis labels
    x_labels = [f"{i+1}" for i in range(len(ret_list))[::-1]]

    # Plot each list
    plt.figure(figsize=(10.5, 6))
    plt.plot(x_labels, ret_list, marker='o', color='midnightblue', linestyle='-', label='Before')
    plt.plot(x_labels, ret_list_erased, marker='^', color='darkorange', linestyle='-', label='After')
    
    # Adding labels and legend
    plt.title('Retrieval Before vs. After LEACE', fontsize=20)
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Top k', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid for better readability

    # Save the plot
    plt.tight_layout()
    plt.savefig('top_k_retrieval.png')
    plt.close()

    ### Get exact pairs

    total_pairs, total_pairs_erased = [], []

    k_list = list(range(2, max_n_clusters))

    for kr in k_list:
        total_pairs.append(get_exact_pairs(embeddings, kr, text_list_1))
        total_pairs_erased.append(get_exact_pairs(embeddings_erased, kr, text_list_1))

    ln = len(embeddings_1)
    total_pairs_prc = [i / ln for i in total_pairs][:max_n_clusters-1]
    total_pairs_erased_prc = [i / ln for i in total_pairs_erased][:max_n_clusters-1]

    # Create the plot
    plt.figure(figsize=(10.5, 6))
    plt.plot(k_list, total_pairs_prc, marker='.', label='Before Erasure', color='midnightblue')
    plt.plot(k_list, total_pairs_erased_prc, marker='.', label='After Erasure', color='darkorange')

    # Add labels, title, and legend
    plt.xlabel('# of Clusters', fontsize=14)
    plt.ylabel('Percentage of Exact Pairs', fontsize=14)
    plt.title('Percentage of Exact Pairs Before and After Erasure', fontsize=20)
    plt.legend(fontsize=12)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.savefig('exact_pairs.png')
    plt.close()

def run_erasure_two_sources_no_label(text_list_1, text_list_2, erase_all=True, embedding='mini', top_k_retrieval=20, max_n_clusters=32): 
    
    # Load the sentence embedding model
    model = load_embedding_model(embedding)

    df = pd.DataFrame({
        'text1': text_list_1,
        'text2': text_list_2,
        })

    # Get embeddings for both text columns
    embeddings_text1 = model.encode(df['text1'].tolist(), convert_to_numpy=True)
    embeddings_text2 = model.encode(df['text2'].tolist(), convert_to_numpy=True)

    # Combine all embeddings for clustering
    all_embeddings = np.vstack((embeddings_text1, embeddings_text2))

    # Run KMeans clustering
    km = 2
    kmeans = KMeans(n_clusters=km, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)

    # Assign clusters back to original dataframe
    df['cluster_text1'] = cluster_labels[:len(df)]
    df['cluster_text2'] = cluster_labels[len(df):]

    # Count non-exact pairs
    df['not_exact_pair'] = df['cluster_text1'] != df['cluster_text2']

    df_ne = df[df['not_exact_pair']]
    df_ep = df[df['not_exact_pair'] == False]
    embeddings_text1_ep = embeddings_text1[df['not_exact_pair'] == False]
    embeddings_text2_ep = embeddings_text2[df['not_exact_pair'] == False]

    # Combine texts and label them
    text_list_1_erasure = df_ne['text1'].to_list()
    text_list_2_erasure = df_ne['text2'].to_list()
    text_list_erasure = text_list_1_erasure + text_list_2_erasure
    labels_erasure = df_ne['cluster_text1'].to_list() + df_ne['cluster_text2'].to_list()

    text_list_1 = df['text1'].to_list()
    text_list_2 = df['text2'].to_list()
    text_list = text_list_1 + text_list_2
    labels = ['source_1'] * len(text_list_1) + ['source_2'] * len(text_list_2)

    # Compute embeddings
    embeddings_erasure = model.encode(text_list_erasure)
    embeddings_erasure = torch.from_numpy(embeddings_erasure)
    embeddings = torch.from_numpy(all_embeddings)

    # Convert string labels to numeric
    numeric_labels_erasure = torch.tensor(labels_erasure)

    # Run LEACE erasure
    eraser = LeaceEraser.fit(embeddings_erasure, numeric_labels_erasure)
    if erase_all:
        embeddings_erased = eraser(embeddings)
        embeddings_erased_1 = embeddings_erased[:len(text_list_1)]
        embeddings_erased_2 = embeddings_erased[len(text_list_1):]
    else:
        embeddings_erased = eraser(embeddings_erasure)
        embeddings_erased_1 = embeddings_erased[:len(text_list_1_erasure)]
        embeddings_erased_2 = embeddings_erased[len(text_list_1_erasure):]
        embeddings_erased_1 = np.vstack((embeddings_text1_ep, embeddings_erased_1))
        embeddings_erased_2 = np.vstack((embeddings_text2_ep, embeddings_erased_2))
        embeddings_erased = np.vstack((embeddings_erased_1, embeddings_erased_2))

    ### Generate top k rankings

    ret_list, ret_list_erased = [], []
    for tk in range(top_k_retrieval, 0, -1):

        top_k_results = is_top_k_match(embeddings_text1, embeddings_text2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list.append(prc)

        top_k_results = is_top_k_match(embeddings_erased_1, embeddings_erased_2, k=tk)
        prc = sum(top_k_results) / len(top_k_results)
        ret_list_erased.append(prc)

    # X-axis labels
    x_labels = [f"{i+1}" for i in range(len(ret_list))[::-1]]

    # Plot each list
    plt.figure(figsize=(10.5, 6))
    plt.plot(x_labels, ret_list, marker='o', color='midnightblue', linestyle='-', label='Before')
    plt.plot(x_labels, ret_list_erased, marker='^', color='darkorange', linestyle='-', label='After')

    # Adding labels and legend
    plt.title('Retrieval Before vs. After LEACE', fontsize=20)
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Top k', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Add grid for better readability

    # Save the plot
    plt.tight_layout()
    plt.savefig('top_k_retrieval.png')
    plt.close()

    ### Get exact pairs

    total_pairs, total_pairs_erased = [], []

    k_list = list(range(2, max_n_clusters))

    for kr in k_list:
        total_pairs.append(get_exact_pairs(all_embeddings, kr, text_list_1))
        total_pairs_erased.append(get_exact_pairs(embeddings_erased, kr, text_list_1))

    ln = len(embeddings_text1)
    total_pairs_prc = [i / ln for i in total_pairs][:max_n_clusters-1]
    total_pairs_erased_prc = [i / ln for i in total_pairs_erased][:max_n_clusters-1]

    # Create the plot
    plt.figure(figsize=(10.5, 6))
    plt.plot(k_list, total_pairs_prc, marker='.', label='Before Erasure', color='midnightblue')
    plt.plot(k_list, total_pairs_erased_prc, marker='.', label='After Erasure', color='darkorange')

    # Add labels, title, and legend
    plt.xlabel('# of Clusters', fontsize=14)
    plt.ylabel('Percentage of Exact Pairs', fontsize=14)
    plt.title('Percentage of Exact Pairs Before and After Erasure', fontsize=20)
    plt.legend(fontsize=12)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.savefig('exact_pairs.png')
    plt.close()
