import requests
import numpy as np
import time
from typing import List, Optional, Callable


def generate_jina_embeddings(texts: List[str], api_key: str, logger_callback: Callable[[str], None]) -> Optional[np.ndarray]:
    """
    Generate embeddings using Jina AI API with batching support.
    
    Args:
        texts: List of text strings to embed
        api_key: Jina API key
        logger_callback: Function to log messages to UI
        
    Returns:
        NumPy array of embeddings or None if failed
    """
    if not api_key:
        logger_callback("ERROR: Jina API key not provided")
        return None
    
    if not texts:
        logger_callback("ERROR: No texts provided for embedding")
        return None
    
    # Debug API key format (show only first and last few characters for security)
    if len(api_key) > 10:
        masked_key = api_key[:8] + "..." + api_key[-4:]
        logger_callback(f"Using API key: {masked_key}")
    else:
        logger_callback("API key seems too short")
    
    # Configuration
    BATCH_SIZE = 20  # Reduced from 128 to avoid timeouts
    API_URL = "https://api.jina.ai/v1/embeddings"
    
    # Check for empty texts
    non_empty_texts = [text for text in texts if text and text.strip()]
    if len(non_empty_texts) < len(texts):
        logger_callback(f"WARNING: Found {len(texts) - len(non_empty_texts)} empty texts out of {len(texts)} total")
    
    if not non_empty_texts:
        logger_callback("ERROR: All texts are empty")
        return None
    
    # Calculate batches
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    logger_callback(f"Starting Jina embedding generation for {len(texts)} texts in {total_batches} batch(es)")
    
    all_embeddings = []
    
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_texts = texts[i:i + BATCH_SIZE]
            
            logger_callback(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            # Prepare payload for Jina API
            payload = {
                "model": "jina-embeddings-v4",
                "task": "text-matching",
                "truncate": True,  # Let Jina API handle truncation automatically
                "input": [{"text": text} for text in batch_texts]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Retry logic for timeouts
            max_retries = 2
            batch_success = False
            
            for retry in range(max_retries + 1):
                try:
                    if retry > 0:
                        logger_callback(f"Retry {retry}/{max_retries} for batch {batch_num}")
                        time.sleep(5)  # Wait before retry
                    
                    # Send request to Jina API with increased timeout
                    response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract embeddings from response
                        if "data" in result:
                            batch_embeddings = []
                            for item in result["data"]:
                                if "embedding" in item:
                                    batch_embeddings.append(item["embedding"])
                            
                            if len(batch_embeddings) == len(batch_texts):
                                all_embeddings.extend(batch_embeddings)
                                logger_callback(f"Batch {batch_num} completed successfully")
                                batch_success = True
                                break  # Success, exit retry loop
                            else:
                                logger_callback(f"ERROR: Batch {batch_num}: Mismatch in embedding count (expected {len(batch_texts)}, got {len(batch_embeddings)})")
                                return None
                        else:
                            logger_callback(f"ERROR: Batch {batch_num}: No 'data' field in response")
                            return None
                            
                    elif response.status_code == 429:
                        logger_callback(f"WARNING: Batch {batch_num}: Rate limit hit, waiting 30 seconds...")
                        time.sleep(30)
                        continue  # Retry immediately
                        
                    else:
                        logger_callback(f"ERROR: Batch {batch_num}: API error {response.status_code}")
                        logger_callback(f"ERROR: Response: {response.text[:200]}...")
                        if retry == max_retries:
                            return None
                        continue  # Try again
                        
                except requests.exceptions.Timeout:
                    logger_callback(f"WARNING: Batch {batch_num}: Request timeout (attempt {retry + 1})")
                    if retry == max_retries:
                        logger_callback(f"ERROR: Batch {batch_num}: All retry attempts failed")
                        return None
                    continue  # Try again
                    
                except requests.exceptions.RequestException as e:
                    logger_callback(f"ERROR: Batch {batch_num}: Request error: {str(e)}")
                    if retry == max_retries:
                        return None
                    continue  # Try again
            
            if not batch_success:
                logger_callback(f"ERROR: Batch {batch_num}: Failed after all retries")
                return None
            
            # Longer delay between batches to be respectful to the API
            if batch_num < total_batches:
                time.sleep(2)
                logger_callback(f"Waiting 2 seconds before next batch...")
    
    except Exception as e:
        logger_callback(f"ERROR: Unexpected error during embedding generation: {str(e)}")
        return None
    
    # Convert to NumPy array
    if all_embeddings:
        try:
            embeddings_array = np.array(all_embeddings)
            logger_callback(f"Successfully generated {embeddings_array.shape[0]} embeddings with {embeddings_array.shape[1]} dimensions")
            return embeddings_array
        except Exception as e:
            logger_callback(f"ERROR: Error converting embeddings to NumPy array: {str(e)}")
            return None
    else:
        logger_callback("ERROR: No embeddings were generated")
        return None

def vectorize_urls_and_content(urls, content_dict=None, use_url_paths=False, use_content=True, url_weight=0.0, 
                             api_key=None, logger_callback=None):
    """
    Legacy wrapper function for compatibility with existing code.
    Now uses Jina embeddings for content-only analysis.
    
    Args:
        urls: List of URLs
        content_dict: Dictionary mapping URLs to content
        use_url_paths: Ignored (legacy parameter)
        use_content: Should be True for Jina embedding mode
        url_weight: Ignored (legacy parameter) 
        api_key: Jina API key
        logger_callback: Function to log messages
        
    Returns:
        tuple: (URLs, empty_paths, processed_contents, None, embeddings_matrix)
    """
    if not use_content or not content_dict:
        if logger_callback:
            logger_callback("ERROR: Content analysis disabled or no content provided. Jina embeddings require content.")
        # Return dummy data for compatibility
        dummy_matrix = np.eye(len(urls))
        empty_paths = [""] * len(urls)
        empty_contents = [""] * len(urls)
        return urls, empty_paths, empty_contents, None, dummy_matrix
    
    # Prepare raw content for Jina embeddings (API will handle truncation)
    raw_contents = []
    for url in urls:
        if url in content_dict:
            # Use raw content to preserve diacritics and multilingual characters
            raw_content = str(content_dict[url]) if content_dict[url] else ""
            raw_contents.append(raw_content)
        else:
            raw_contents.append("")
    
    # Generate embeddings using Jina
    if api_key and logger_callback:
        embeddings_matrix = generate_jina_embeddings(raw_contents, api_key, logger_callback)
        
        if embeddings_matrix is not None:
            # Return in expected format
            empty_paths = [""] * len(urls)  # No URL path processing in new version
            return urls, empty_paths, raw_contents, None, embeddings_matrix
        else:
            # Fallback to dummy matrix if embedding generation failed
            if logger_callback:
                logger_callback("WARNING: Falling back to dummy matrix due to embedding generation failure")
            dummy_matrix = np.eye(len(urls))
            empty_paths = [""] * len(urls)
            return urls, empty_paths, raw_contents, None, dummy_matrix
    else:
        if logger_callback:
            logger_callback("ERROR: Missing API key or logger callback for Jina embeddings")
        # Return dummy data
        dummy_matrix = np.eye(len(urls))
        empty_paths = [""] * len(urls)
        return urls, empty_paths, raw_contents, None, dummy_matrix

# Test function for the new Jina embedding functionality
def test_jina_embeddings():
    """Test function for Jina embeddings (requires valid API key)"""
    test_texts = [
        "This is a product page about laptops and computers.",
        "Learn more about our company history and mission.",
        "Contact us for customer support and inquiries.",
        "Blog post about the latest technology trends.",
        "Pricing information for our software services."
    ]
    
    def dummy_logger(message):
        print(f"LOG: {message}")
    
    # Try to get API key from environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        api_key = "your_jina_api_key_here"  # Replace with actual key for testing
    
    print("Testing Jina embeddings...")
    embeddings = generate_jina_embeddings(test_texts, api_key, dummy_logger)
    
    if embeddings is not None:
        print(f"Success! Generated embeddings with shape: {embeddings.shape}")
    else:
        print("Failed to generate embeddings")

if __name__ == "__main__":
    test_jina_embeddings()