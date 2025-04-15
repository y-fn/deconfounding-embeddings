import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from concept_erasure import LeaceEraser

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

def run_erasure_two_sources(text_list_1, text_list_2, embedding='mpnet', k=5, top_k_retrieval=3, max_n_clusters=8): 
    # Load the sentence embedding model
    if embedding == 'mpnet':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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
    plt.scatter(source_2_pca[:, 0], source_2_pca[:, 1], color='darkorange', label='Wiki', alpha=0.7, s=15)  # Smaller points
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
    plt.title('PCA of Embeddings Before Erasure', fontsize=20)
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
    total_pairs_prc = [i / ln for i in total_pairs][:127]
    total_pairs_erased_prc = [i / ln for i in total_pairs_erased][:127]

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
