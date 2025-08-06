import numpy as np
import json
import re
from typing import List, Dict, Optional, Tuple, Callable
from urllib.parse import urlparse
from sklearn.cluster import DBSCAN


def clean_url_for_display(url: str) -> str:
    """
    Clean URL for display by removing protocol and www prefix.
    
    Args:
        url (str): Original URL
        
    Returns:
        str: Cleaned URL (e.g., 'https://www.example.com/path' -> 'example.com/path')
    """
    if not url:
        return url
    
    # Remove protocol
    cleaned = re.sub(r'^https?://', '', url)
    
    # Remove www. prefix
    cleaned = re.sub(r'^www\.', '', cleaned)
    
    return cleaned


def perform_topic_clustering(embedding_matrix: np.ndarray, eps: float = 0.3, min_samples: int = 3) -> np.ndarray:
    """
    Perform DBSCAN clustering on high-dimensional embeddings.
    
    Args:
        embedding_matrix (np.ndarray): High-dimensional embeddings matrix
        eps (float): Maximum distance between samples for clustering (default: 0.3)
        min_samples (int): Minimum samples in neighborhood for core point (default: 3)
        
    Returns:
        np.ndarray: Cluster labels (-1 for noise, 0+ for clusters)
    """
    if embedding_matrix is None or embedding_matrix.shape[0] < 2:
        # Not enough data for clustering
        return np.array([0] * embedding_matrix.shape[0] if embedding_matrix is not None else [])
    
    # Use DBSCAN with cosine distance (standard for embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = clustering.fit_predict(embedding_matrix)
    
    return cluster_labels


def prepare_cluster_data_for_llm(urls: List[str], cluster_labels: np.ndarray) -> Tuple[Dict[int, List[str]], str]:
    """
    Prepare clustered URL data for LLM prompt.
    
    Args:
        urls (List[str]): List of URLs
        cluster_labels (np.ndarray): Cluster labels from DBSCAN
        
    Returns:
        Tuple[Dict[int, List[str]], str]: (cluster_dict, formatted_string)
    """
    cluster_dict = {}
    
    # Group URLs by cluster (exclude noise: label -1)
    for i, (url, label) in enumerate(zip(urls, cluster_labels)):
        if label != -1:  # Exclude noise
            if label not in cluster_dict:
                cluster_dict[label] = []
            cleaned_url = clean_url_for_display(url)
            cluster_dict[label].append(cleaned_url)
    
    # Format for LLM prompt
    formatted_parts = []
    for cluster_id in sorted(cluster_dict.keys()):
        formatted_parts.append(f"Cluster {cluster_id}:")
        for url in cluster_dict[cluster_id]:
            formatted_parts.append(f"- {url}")
        formatted_parts.append("")  # Empty line between clusters
    
    formatted_string = "\n".join(formatted_parts).strip()
    
    return cluster_dict, formatted_string


def create_cluster_naming_prompt(cluster_data: str, total_url_count: int, cluster_count: int) -> str:
    """
    Create the prompt for LLM cluster naming.
    
    Args:
        cluster_data (str): Formatted cluster data
        total_url_count (int): Total number of URLs
        cluster_count (int): Number of clusters found
        
    Returns:
        str: Complete prompt for LLM
    """
    prompt_template = """You are an expert SEO strategist specializing in information architecture. Your task is to analyze groups of URLs that have been algorithmically clustered based on the semantic meaning of their content. For each cluster, you will provide a concise, descriptive name (2-4 words) that best represents the core topic of that group.

**Instructions:**
1. Analyze the provided URL paths within each cluster.
2. Identify the common theme or topic for each cluster.
3. Assign a short, accurate name to each cluster.
4. **Language Rule:** You MUST assign the name in the primary language detected within the URLs for that specific cluster.
5. **Output Format:** Provide your response as a single, valid JSON object. Each key in the JSON should be the cluster identifier (e.g., "Cluster 0"), and the corresponding value should be the name you have assigned. Do not include any other text or explanation outside of the JSON object.

**Analysis Data:**
- Total URLs Analyzed: {total_url_count}
- Total Clusters Found: {cluster_count}

Here are the clusters and their associated URLs:

{cluster_data}"""

    return prompt_template.format(
        total_url_count=total_url_count,
        cluster_count=cluster_count,
        cluster_data=cluster_data
    )


def parse_cluster_names_from_llm_response(llm_response: str, cluster_dict: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Parse cluster names from LLM JSON response with error handling.
    
    Args:
        llm_response (str): Raw response from LLM
        cluster_dict (Dict[int, List[str]]): Original cluster dictionary
        
    Returns:
        Dict[int, str]: Mapping from cluster ID to cluster name
    """
    cluster_names = {}
    
    try:
        # Try to parse JSON from response
        # First, try to extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = llm_response.strip()
        
        parsed_response = json.loads(json_str)
        
        # Extract cluster names
        for cluster_id in cluster_dict.keys():
            cluster_key = f"Cluster {cluster_id}"
            if cluster_key in parsed_response:
                cluster_names[cluster_id] = parsed_response[cluster_key]
            else:
                # Fallback to default name
                cluster_names[cluster_id] = f"Topic Cluster {cluster_id}"
        
    except (json.JSONDecodeError, AttributeError) as e:
        # JSON parsing failed, use default names
        print(f"Warning: Failed to parse LLM response for cluster naming: {e}")
        for cluster_id in cluster_dict.keys():
            cluster_names[cluster_id] = f"Topic Cluster {cluster_id}"
    
    return cluster_names


def get_cluster_analysis_summary(urls: List[str], cluster_labels: np.ndarray, cluster_names: Dict[int, str]) -> List[Dict]:
    """
    Generate summary data for cluster analysis tab.
    
    Args:
        urls (List[str]): List of URLs
        cluster_labels (np.ndarray): Cluster labels
        cluster_names (Dict[int, str]): Cluster names mapping
        
    Returns:
        List[Dict]: Summary data for each cluster
    """
    summary_data = []
    
    # Count URLs per cluster
    cluster_counts = {}
    cluster_examples = {}
    
    for url, label in zip(urls, cluster_labels):
        if label != -1:  # Exclude noise
            if label not in cluster_counts:
                cluster_counts[label] = 0
                cluster_examples[label] = []
            cluster_counts[label] += 1
            if len(cluster_examples[label]) < 3:  # Keep max 3 examples
                cluster_examples[label].append(clean_url_for_display(url))
    
    # Create summary for each cluster
    for cluster_id in sorted(cluster_counts.keys()):
        cluster_name = cluster_names.get(cluster_id, f"Topic Cluster {cluster_id}")
        summary_data.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'url_count': cluster_counts[cluster_id],
            'example_urls': cluster_examples[cluster_id]
        })
    
    # Add noise/outliers if present
    noise_count = sum(1 for label in cluster_labels if label == -1)
    if noise_count > 0:
        summary_data.append({
            'cluster_id': -1,
            'cluster_name': 'Outliers/Noise',
            'url_count': noise_count,
            'example_urls': ['(Individual pages with unique topics)']
        })
    
    return summary_data


def test_clustering():
    """Test function for clustering module."""
    # Sample data
    urls = [
        'https://example.com/products/laptop-1',
        'https://example.com/products/laptop-2', 
        'https://example.com/blog/tech-news',
        'https://example.com/blog/industry-update',
        'https://example.com/about-us',
        'https://example.com/contact'
    ]
    
    # Mock embeddings (in real use, these come from Jina)
    np.random.seed(42)
    embeddings = np.random.rand(6, 10)
    
    # Test clustering
    cluster_labels = perform_topic_clustering(embeddings, eps=0.5, min_samples=2)
    print(f"Cluster labels: {cluster_labels}")
    
    # Test data preparation
    cluster_dict, formatted_data = prepare_cluster_data_for_llm(urls, cluster_labels)
    print(f"Cluster dictionary: {cluster_dict}")
    print(f"Formatted data:\n{formatted_data}")
    
    # Test prompt creation
    prompt = create_cluster_naming_prompt(formatted_data, len(urls), len(cluster_dict))
    print(f"\nPrompt length: {len(prompt)} characters")
    
    # Test LLM response parsing
    mock_llm_response = '{"Cluster 0": "Product Pages", "Cluster 1": "Blog Content"}'
    cluster_names = parse_cluster_names_from_llm_response(mock_llm_response, cluster_dict)
    print(f"Parsed cluster names: {cluster_names}")
    
    # Test summary
    summary = get_cluster_analysis_summary(urls, cluster_labels, cluster_names)
    print(f"Cluster summary: {summary}")


if __name__ == "__main__":
    test_clustering()