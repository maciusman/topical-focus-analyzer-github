import os
import time
import requests
import numpy as np
import chromadb
from typing import List, Dict, Optional, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_jina_embeddings(texts: List[str], api_key: str, logger_callback: Callable[[str], None]) -> Optional[np.ndarray]:
    """
    Generate embeddings using Jina AI API with batching support.
    Reuses the existing logic from simple_vectorizer.py for consistency.
    """
    if not api_key:
        logger_callback("ERROR: Jina API key not provided")
        return None
    
    if not texts:
        logger_callback("ERROR: No texts provided for embedding")
        return None
    
    # Configuration
    BATCH_SIZE = 20  # Conservative batch size
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
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Skip empty texts in this batch
            batch_texts = [text if text and text.strip() else "empty content" for text in batch_texts]
            
            logger_callback(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_texts)} texts)")
            
            # API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "jina-embeddings-v4",
                "task": "text-matching",
                "truncate": True,
                "input": [{"text": text} for text in batch_texts]
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data:
                            batch_embeddings = [item['embedding'] for item in data['data']]
                            all_embeddings.extend(batch_embeddings)
                            logger_callback(f"Batch {batch_idx + 1} successful")
                            break
                        else:
                            logger_callback(f"ERROR: No data in response for batch {batch_idx + 1}")
                            return None
                    else:
                        logger_callback(f"ERROR: API returned status {response.status_code} for batch {batch_idx + 1}")
                        if attempt < max_retries - 1:
                            logger_callback(f"Retrying batch {batch_idx + 1} (attempt {attempt + 2})")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            return None
                            
                except requests.exceptions.RequestException as e:
                    logger_callback(f"ERROR: Request failed for batch {batch_idx + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        logger_callback(f"Retrying batch {batch_idx + 1} (attempt {attempt + 2})")
                        time.sleep(2 ** attempt)
                    else:
                        return None
            
            # Rate limiting between batches
            if batch_idx < total_batches - 1:
                time.sleep(1)
    
    except Exception as e:
        logger_callback(f"ERROR: Unexpected error during embedding generation: {str(e)}")
        return None
    
    logger_callback(f"Successfully generated {len(all_embeddings)} embeddings")
    return np.array(all_embeddings)


def create_semantic_index(
    content_data: List[Dict], 
    db_path: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    jina_api_key: str,
    logger_callback: Callable[[str], None]
) -> bool:
    """
    Creates and persists a ChromaDB vector database from the provided content data.
    
    Args:
        content_data: List of dicts with 'url' and 'content' keys
        db_path: Directory path where ChromaDB will be stored
        collection_name: Name of the ChromaDB collection
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        jina_api_key: Jina AI API key for embeddings
        logger_callback: Function to log messages to UI
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Ensure directory exists and initialize ChromaDB client
        logger_callback(f"Initializing database client at: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        
        # Delete existing collection if it exists (for fresh indexing)
        try:
            client.delete_collection(name=collection_name)
            logger_callback(f"Deleted existing collection: '{collection_name}'")
        except:
            pass  # Collection doesn't exist, which is fine
            
        collection = client.create_collection(name=collection_name)
        logger_callback(f"Created new collection: '{collection_name}'")

        # Step 2: Chunk documents
        logger_callback(f"Starting content chunking (size: {chunk_size}, overlap: {chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        all_metadatas = []
        
        for item in content_data:
            url = item.get('url', 'unknown_url')
            content = item.get('content', '')
            
            if content and content.strip():
                # Split content into chunks
                chunks = text_splitter.split_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            'source_url': url,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks)
                        })

        if not all_chunks:
            logger_callback("Error: No content to chunk. Indexing aborted.")
            return False
            
        logger_callback(f"Created {len(all_chunks)} text chunks from {len(content_data)} documents")

        # Step 3: Generate embeddings
        logger_callback("Generating embeddings for all chunks...")
        embeddings = generate_jina_embeddings(all_chunks, jina_api_key, logger_callback)
        if embeddings is None:
            logger_callback("Error: Failed to generate embeddings. Indexing aborted.")
            return False

        # Step 4: Add data to ChromaDB collection in batches to avoid size limits
        BATCH_SIZE_CHROMA = 1000  # A safe batch size for ChromaDB
        
        logger_callback(f"Adding {len(all_chunks)} chunks to the local vector store in batches...")
        
        # Generate IDs for all chunks first
        ids = [f"{item['source_url']}_{item['chunk_index']}" for item in all_metadatas]
        
        # Convert embeddings to list once
        embeddings_list = embeddings.tolist()
        
        total_chunks = len(all_chunks)
        for i in range(0, total_chunks, BATCH_SIZE_CHROMA):
            # Determine the end of the batch
            end_index = min(i + BATCH_SIZE_CHROMA, total_chunks)
            
            # Log progress for the user
            logger_callback(f"Processing batch {i // BATCH_SIZE_CHROMA + 1}/{(total_chunks + BATCH_SIZE_CHROMA - 1) // BATCH_SIZE_CHROMA} (chunks {i+1}-{end_index})")
            
            # Get the slice for the current batch
            batch_documents = all_chunks[i:end_index]
            batch_embeddings = embeddings_list[i:end_index]
            batch_metadatas = all_metadatas[i:end_index]
            batch_ids = ids[i:end_index]
            
            # Call collection.add() for the current batch
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        logger_callback("Success! All batches have been added to the semantic index.")
        return True

    except Exception as e:
        logger_callback(f"An unexpected error occurred during index creation: {str(e)}")
        return False


def query_semantic_index(
    query_text: str,
    db_path: str,
    collection_name: str,
    jina_api_key: str,
    top_k: int,
    reranker_model_name: str,
    logger_callback: Callable[[str], None]
) -> List[Dict]:
    """
    Queries an existing index for a SINGLE query, re-ranks the results, and returns them.
    
    Args:
        query_text: The search query
        db_path: Directory path where ChromaDB is stored
        collection_name: Name of the ChromaDB collection
        jina_api_key: Jina AI API key
        top_k: Number of results to retrieve before re-ranking
        reranker_model_name: Jina reranker model to use
        logger_callback: Function to log messages
        
    Returns:
        List of dicts with ranked results
    """
    try:
        if not query_text or not query_text.strip():
            logger_callback("Error: Empty query provided")
            return []
            
        # Step 1: Load ChromaDB collection
        if not os.path.exists(db_path):
            logger_callback(f"Error: Database path does not exist: {db_path}")
            return []
            
        client = chromadb.PersistentClient(path=db_path)
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            logger_callback(f"Error: Collection '{collection_name}' not found: {str(e)}")
            return []

        # Step 2: Generate query embedding
        logger_callback("Generating query embedding...")
        query_embedding = generate_jina_embeddings([query_text], jina_api_key, logger_callback)
        if query_embedding is None:
            logger_callback("Error: Failed to generate query embedding")
            return []

        # Step 3: Retrieve similar documents
        logger_callback(f"Searching for top {top_k} similar documents...")
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, collection.count())
        )
        
        if not results['documents'][0]:
            logger_callback("No documents found matching the query")
            return []
            
        # Step 4: Prepare documents for re-ranking
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        logger_callback(f"Found {len(documents)} candidate documents. Applying re-ranking...")
        
        # Step 5: Re-rank using Jina Reranker API
        reranked_results = []
        try:
            rerank_url = "https://api.jina.ai/v1/rerank"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {jina_api_key}"
            }
            
            payload = {
                "model": reranker_model_name,
                "query": query_text,
                "documents": documents,
                "top_n": min(len(documents), top_k)
            }
            
            response = requests.post(rerank_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                rerank_data = response.json()
                
                for i, result in enumerate(rerank_data.get('results', [])):
                    doc_index = result.get('index', i)
                    relevance_score = result.get('relevance_score', 0.0)
                    
                    if doc_index < len(documents):
                        reranked_results.append({
                            'rank': i + 1,
                            'relevance_score': relevance_score,
                            'content': documents[doc_index],
                            'source_url': metadatas[doc_index].get('source_url', 'unknown'),
                            'chunk_index': metadatas[doc_index].get('chunk_index', 0),
                            'original_distance': distances[doc_index] if doc_index < len(distances) else 1.0
                        })
                        
                logger_callback(f"Re-ranking completed. Returning {len(reranked_results)} results.")
                
            else:
                logger_callback(f"Re-ranking API error (status {response.status_code}). Using original ranking.")
                # Fallback to original ranking
                for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                    reranked_results.append({
                        'rank': i + 1,
                        'relevance_score': 1.0 - dist,  # Convert distance to similarity
                        'content': doc,
                        'source_url': meta.get('source_url', 'unknown'),
                        'chunk_index': meta.get('chunk_index', 0),
                        'original_distance': dist
                    })
                    
        except Exception as e:
            logger_callback(f"Re-ranking failed: {str(e)}. Using original ranking.")
            # Fallback to original ranking
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                reranked_results.append({
                    'rank': i + 1,
                    'relevance_score': 1.0 - dist,
                    'content': doc,
                    'source_url': meta.get('source_url', 'unknown'),
                    'chunk_index': meta.get('chunk_index', 0),
                    'original_distance': dist
                })
        
        return reranked_results
        
    except Exception as e:
        logger_callback(f"An unexpected error occurred during query: {str(e)}")
        return []


def check_index_exists(db_path: str, collection_name: str) -> bool:
    """
    Check if a semantic index exists at the given path and collection name.
    
    Args:
        db_path: Directory path where ChromaDB should be stored
        collection_name: Name of the ChromaDB collection
        
    Returns:
        bool: True if index exists and is accessible, False otherwise
    """
    try:
        if not os.path.exists(db_path):
            return False
            
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        return collection.count() > 0
        
    except Exception:
        return False


def get_index_stats(db_path: str, collection_name: str) -> Dict:
    """
    Get statistics about an existing semantic index.
    
    Args:
        db_path: Directory path where ChromaDB is stored
        collection_name: Name of the ChromaDB collection
        
    Returns:
        Dict with statistics or empty dict if error
    """
    try:
        if not os.path.exists(db_path):
            return {}
            
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        
        count = collection.count()
        if count == 0:
            return {}
            
        # Get a sample to analyze metadata
        sample = collection.get(limit=min(100, count))
        
        # Count unique URLs
        unique_urls = set()
        for metadata in sample['metadatas']:
            if metadata and 'source_url' in metadata:
                unique_urls.add(metadata['source_url'])
        
        return {
            'total_chunks': count,
            'unique_urls': len(unique_urls),
            'sample_size': len(sample['metadatas']) if sample['metadatas'] else 0
        }
        
    except Exception:
        return {}