# --- START OF FILE multi_sitemap_app.py ---

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from dotenv import load_dotenv
import numpy as np
import pickle  # Added for saving/loading
import base64 # Added for CSV download link

# Import our custom modules
# Ensure these modules exist in a 'modules' directory or adjust path as needed
try:
    from modules.content_extractor import batch_extract_content, extract_main_content
    from modules.simple_vectorizer import vectorize_urls_and_content, generate_jina_embeddings
    from modules.dimensionality_reducer import reduce_dimensions_and_find_centroid
    from modules.analyzer import calculate_metrics, find_potential_duplicates
    from modules.semantic_search_engine import (
        create_semantic_index, query_semantic_index, 
        check_index_exists, get_index_stats
    )
    from modules.llm_summarizer import get_gemini_summary, fetch_openrouter_models, generate_summary_with_openrouter, generate_cluster_names_with_openrouter
    from modules.clustering import (perform_topic_clustering, prepare_cluster_data_for_llm, 
                                   create_cluster_naming_prompt, parse_cluster_names_from_llm_response,
                                   get_cluster_analysis_summary)
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure the 'modules' directory and required files exist.")
    st.stop()

# Load environment variables
load_dotenv()

# Function for interactive clustering (Phase 2)
def perform_interactive_cluster_analysis(eps: float, min_samples: int):
    """
    Perform interactive clustering analysis using stored embeddings.
    This function encapsulates the clustering logic for Phase 2.
    
    Args:
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN min_samples parameter
    
    Returns:
        bool: True if successful, False otherwise
    """
    if st.session_state.embeddings_matrix is None:
        st.error("No embeddings matrix found. Please run Phase 1 analysis first.")
        return False
    
    if st.session_state.results_df is None:
        st.error("No analysis results found. Please run Phase 1 analysis first.")
        return False
    
    # Get API key and model for LLM naming
    openrouter_api_key = st.session_state.input_openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    selected_model = st.session_state.input_selected_model
    
    try:
        # Perform DBSCAN clustering on stored embeddings
        cluster_labels = perform_topic_clustering(
            embedding_matrix=st.session_state.embeddings_matrix,
            eps=eps,
            min_samples=min_samples
        )
        
        st.session_state.cluster_labels = cluster_labels
        
        # Count clusters (excluding noise: -1)
        unique_clusters = set(cluster_labels)
        noise_count = sum(1 for label in cluster_labels if label == -1)
        cluster_count = len([c for c in unique_clusters if c != -1])
        
        st.success(f"🏷️ Found {cluster_count} topic clusters, {noise_count} outliers")
        
        # Prepare data for LLM naming (only if we have clusters and API key)
        if cluster_count > 0 and openrouter_api_key and selected_model:
            with st.spinner("🤖 Generating cluster names with AI..."):
                cluster_dict, formatted_cluster_data = prepare_cluster_data_for_llm(
                    urls=st.session_state.urls,
                    cluster_labels=cluster_labels
                )
                
                # Create prompt for LLM
                cluster_naming_prompt = create_cluster_naming_prompt(
                    cluster_data=formatted_cluster_data,
                    total_url_count=len(st.session_state.urls),
                    cluster_count=cluster_count
                )
                
                # Send to LLM for naming
                try:
                    llm_response = generate_cluster_names_with_openrouter(
                        api_key=openrouter_api_key,
                        model_id=selected_model,
                        cluster_prompt=cluster_naming_prompt,
                        logger_callback=None
                    )
                    
                    if llm_response:
                        cluster_names = parse_cluster_names_from_llm_response(llm_response, cluster_dict)
                        st.session_state.cluster_names = cluster_names
                        st.success(f"✅ Successfully named {len(cluster_names)} clusters with AI")
                    else:
                        # Fallback to default names
                        st.session_state.cluster_names = {cid: f"Topic Cluster {cid}" for cid in cluster_dict.keys()}
                        st.warning("⚠️ Using default cluster names due to API response issues")
                        
                except Exception as e:
                    st.warning(f"⚠️ Error during cluster naming: {str(e)}")
                    # Fallback to default names
                    cluster_dict = {}
                    for i, (url, label) in enumerate(zip(st.session_state.urls, cluster_labels)):
                        if label != -1 and label not in cluster_dict:
                            cluster_dict[label] = []
                    st.session_state.cluster_names = {cid: f"Topic Cluster {cid}" for cid in cluster_dict.keys()}
        else:
            # No clusters found or no API key - use default names
            if cluster_count > 0:
                cluster_dict = {}
                for i, (url, label) in enumerate(zip(st.session_state.urls, cluster_labels)):
                    if label != -1 and label not in cluster_dict:
                        cluster_dict[label] = []
                st.session_state.cluster_names = {cid: f"Topic Cluster {cid}" for cid in cluster_dict.keys()}
                if not openrouter_api_key or not selected_model:
                    st.info("ℹ️ Using default cluster names (no API key or model selected)")
            else:
                st.session_state.cluster_names = {}
                st.info("ℹ️ No topic clusters found - all content appears unique")
        
        # Update results_df with cluster information
        if len(cluster_labels) == len(st.session_state.results_df):
            cluster_names_list = []
            for label in cluster_labels:
                if label == -1:
                    cluster_names_list.append("Outliers/Noise")
                else:
                    cluster_name = st.session_state.cluster_names.get(label, f"Topic Cluster {label}")
                    cluster_names_list.append(cluster_name)
            st.session_state.results_df['Topic Cluster'] = cluster_names_list
        
        # Generate cluster analysis summary
        st.session_state.cluster_summary = get_cluster_analysis_summary(
            urls=st.session_state.urls,
            cluster_labels=st.session_state.cluster_labels,
            cluster_names=st.session_state.cluster_names
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error during clustering analysis: {str(e)}")
        return False


# --- Session State Initialization ---
# Initialize session state variables FIRST, before widgets access them
# We check if they exist because loading might populate them before widgets are drawn

# Analysis results
DEFAULT_SESSION_STATE = {
    'sitemaps': None,
    'selected_sitemaps': [],
    'urls': None,
    'url_sources': {},
    'content_dict': None,
    'results_df': None,
    'focus_score': None,
    'radius_score': None,
    'pairwise_distances': None,
    'llm_summary': None,
    'centroid': None,
    'content_for_embedding': None, # List of raw content strings used for embedding generation
    'analysis_loaded': False, # Flag to indicate if state was loaded
    '_loading_in_progress': False, # Temp flag during loading
    'log_messages': [], # New: For detailed logging
    'embeddings_matrix': None, # New: Stored embeddings for interactive clustering
    'cluster_labels': None, # New: Cluster labels from DBSCAN (Phase 2)
    'cluster_names': {}, # New: Mapping from cluster ID to cluster name (Phase 2)
    'cluster_summary': [], # New: Summary data for cluster analysis tab (Phase 2)
    
    # Internal Semantic Search
    'search_db_path': None,
    'search_collection_name': 'default_collection',
    'search_index_exists': False,
    'search_results': None,

    # Input parameters with defaults
    'domain': "",
    'input_include_filters': ["", "", ""], # Use a distinct key for input state storage
    'input_exclude_filters': ["", "", ""], # Use a distinct key for input state storage
    'input_include_logic_any': True,
    'input_analyze_content': True,
    'input_max_workers': 3,
    'input_request_delay': 1.0,
    'input_advanced_extraction': False,
    'input_max_urls': 100,
    'input_perplexity': 15,
    'input_focus_k': 5.0,
    'input_radius_k': 5.0,
    'input_use_ai_summary': True,
    'input_jina_api_key': "", # New: Jina API key
    'input_openrouter_api_key': "", # New: OpenRouter API key
    'input_selected_model': "", # New: Selected AI model
    'input_num_blocks_to_combine': 2 # New: Number of content blocks to combine
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Pickle Load Function ---
def load_analysis_state(uploaded_file):
    """Loads analysis state from a .pkl file."""
    try:
        state = pickle.load(uploaded_file)

        # Restore results state first
        for key in [
            'sitemaps', 'selected_sitemaps', 'urls', 'url_sources', 'content_dict',
            'results_df', 'focus_score', 'radius_score',
            'pairwise_distances', 'llm_summary', 'centroid', 'content_for_embedding',
            'embeddings_matrix', 'cluster_labels', 'cluster_names', 'cluster_summary'
            ]:
            st.session_state[key] = state.get(key, DEFAULT_SESSION_STATE.get(key)) # Use loaded or default

        # Restore input parameters to their specific session state keys
        input_param_keys = [
            'input_domain', 'input_include_filters', 'input_exclude_filters',
            'input_include_logic_any', 'input_analyze_content', 'input_max_workers', 
            'input_request_delay', 'input_advanced_extraction', 'input_max_urls', 'input_perplexity', 'input_focus_k', 
            'input_radius_k', 'input_use_ai_summary', 'input_jina_api_key', 
            'input_openrouter_api_key', 'input_selected_model', 'input_num_blocks_to_combine'
        ]
        for key in input_param_keys:
             # Map loaded 'input_domain' back to 'domain' in session state etc.
             session_state_key = key # Assume mapping like 'input_domain' -> 'input_domain'
             if key == 'input_domain': session_state_key = 'domain' # Special case for domain

             loaded_value = state.get(key, DEFAULT_SESSION_STATE.get(key))

             # Ensure filter lists have correct length
             if key == 'input_include_filters' or key == 'input_exclude_filters':
                  loaded_value = (loaded_value + ["", "", ""])[:3]

             st.session_state[session_state_key] = loaded_value

        st.session_state.analysis_loaded = True # Set flag to prevent re-analysis
        st.success("Analysis state loaded successfully! Sidebar values updated.")
        # No need to rerun here, sidebar widgets will read the updated session state on the *next* natural rerun

        return True

    except pickle.UnpicklingError:
        st.error("Error: Could not load the file. It might be corrupted or not a valid analysis file.")
        st.session_state.analysis_loaded = False # Ensure flag is false on error
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        st.session_state.analysis_loaded = False # Ensure flag is false on error
        return False

# --- Pickle Save Function ---
def save_analysis_state(filename):
    """Saves the current analysis state to a .pkl file."""
    if st.session_state.results_df is None:
        st.warning("No analysis results available to save.")
        return

    try:
        # Gather current input parameters directly from session state (should be up-to-date)
        input_params = {
            'input_domain': st.session_state.domain,
            'input_include_filters': st.session_state.input_include_filters,
            'input_exclude_filters': st.session_state.input_exclude_filters,
            'input_include_logic_any': st.session_state.input_include_logic_any,
            'input_analyze_content': st.session_state.input_analyze_content,
            'input_max_workers': st.session_state.input_max_workers,
            'input_request_delay': st.session_state.input_request_delay,
            'input_max_urls': st.session_state.input_max_urls,
            'input_perplexity': st.session_state.input_perplexity,
            'input_focus_k': st.session_state.input_focus_k,
            'input_radius_k': st.session_state.input_radius_k,
            'input_use_ai_summary': st.session_state.input_use_ai_summary,
            'input_selected_model': st.session_state.input_selected_model,
            # API keys excluded from saving for security
        }

        # Gather results state
        results_state = {
            'sitemaps': st.session_state.sitemaps,
            'selected_sitemaps': st.session_state.selected_sitemaps,
            'urls': st.session_state.urls,
            'url_sources': st.session_state.url_sources,
            'content_dict': st.session_state.content_dict,
            'results_df': st.session_state.results_df,
            'focus_score': st.session_state.focus_score,
            'radius_score': st.session_state.radius_score,
            'pairwise_distances': st.session_state.pairwise_distances,
            'llm_summary': st.session_state.llm_summary,
            'centroid': st.session_state.centroid,
            'content_for_embedding': st.session_state.content_for_embedding,
            'log_messages': st.session_state.log_messages,
            'embeddings_matrix': st.session_state.embeddings_matrix,
            'cluster_labels': st.session_state.cluster_labels,
            'cluster_names': st.session_state.cluster_names,
            'cluster_summary': st.session_state.cluster_summary,
        }

        # Combine parameters and results
        analysis_state = {**input_params, **results_state}

        with open(filename, "wb") as f:
            pickle.dump(analysis_state, f)
        st.success(f"Analysis state successfully saved to `{filename}`")

    except Exception as e:
        st.error(f"Error saving analysis state: {e}")


# Page configuration (Keep as is)
st.set_page_config(
    page_title="Topical Focus Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description (Keep as is)
st.title("🔍 Topical Focus Analyzer")
st.markdown("""
This tool analyzes the topical focus of a website by examining both URL structure and page content.
It visualizes how tightly focused or widely spread the content topics are.
**New:** You can now save and load analysis results using the options in the sidebar.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Load Previous Analysis")
    st.warning("⚠️ Loading .pkl files can be risky. Only load files you trust.")
    uploaded_file = st.file_uploader("Upload .pkl analysis file:", type=["pkl"], key="file_uploader")

    if uploaded_file is not None and not st.session_state._loading_in_progress:
        # Check if this upload is different from the last processed one to avoid loops
        if uploaded_file is not st.session_state.get('_last_uploaded_file', None):
            st.session_state._loading_in_progress = True
            load_analysis_state(uploaded_file)
            st.session_state._last_uploaded_file = uploaded_file # Store reference to prevent reload on same file
            st.session_state._loading_in_progress = False
            st.rerun() # Rerun necessary to reflect loaded state in widgets

    st.divider() # Visual separator

    st.header("Analysis Parameters")

    # API Keys Section
    st.subheader("API Configuration")
    st.session_state.input_jina_api_key = st.text_input(
        "Jina API Key:",
        value="",  # Always start empty for security
        type="password",
        key="jina_api_key_input",
        help="Required for embedding generation. Leave empty to use .env file"
    )
    
    st.session_state.input_openrouter_api_key = st.text_input(
        "OpenRouter API Key:",
        value="",  # Always start empty for security
        type="password",
        key="openrouter_api_key_input",
        help="Required for AI summaries. Leave empty to use .env file"
    )
    
    st.caption("💡 API keys can also be defined in the .env file")

    # Domain input - Reads from and writes to st.session_state.domain
    st.session_state.domain = st.text_input(
        "Enter a domain (e.g., example.com):",
        value=st.session_state.domain, # Read from state
        key="domain_input",
        help="Optional: Used for domain validation of input URLs"
    )
    # Assign to local variable AFTER widget for use in current script run
    domain = st.session_state.domain

    # --- URL Filtering ---
    st.subheader("URL Filtering")
    with st.expander("URL Filters", expanded=True):
        # Include Filters
        st.markdown("**Include URLs containing:**")
        current_includes = st.session_state.input_include_filters[:] # Work with a copy
        for i in range(3):
            current_includes[i] = st.text_input(
                f"Include filter #{i+1}:",
                value=current_includes[i], # Read from list
                key=f"include_{i}"
            )
        st.session_state.input_include_filters = current_includes # Update state list

        # Exclude Filters
        st.markdown("**Exclude URLs containing:**")
        current_excludes = st.session_state.input_exclude_filters[:] # Work with a copy
        for i in range(3):
             current_excludes[i] = st.text_input(
                 f"Exclude filter #{i+1}:",
                 value=current_excludes[i], # Read from list
                 key=f"exclude_{i}"
            )
        st.session_state.input_exclude_filters = current_excludes # Update state list

        # Filter Logic
        filter_logic_index = 1 if not st.session_state.input_include_logic_any else 0
        filter_logic = st.radio(
            "Include filter logic:",
            ["Match ANY filter (OR)", "Match ALL filters (AND)"],
            index=filter_logic_index, # Read from state
            key='filter_logic_radio',
            help="For multiple include filters, choose whether URLs should match any or all of the filters"
        )
        st.session_state.input_include_logic_any = (filter_logic == "Match ANY filter (OR)")

    # --- Content Analysis Options ---
    st.subheader("Content Analysis Options")
    st.session_state.input_analyze_content = st.checkbox(
        "Analyze Page Content (required for Jina embeddings)",
        value=st.session_state.input_analyze_content, # Read from state
        key='analyze_content_cb',
        help="Content analysis is required for the new Jina embedding system"
    )
    # Assign to local variable AFTER widget
    analyze_content = st.session_state.input_analyze_content

    if analyze_content:
        st.info("ℹ️ Using Jina Embeddings v4 for semantic content analysis (content-only mode)")
        
        with st.expander("Content Extraction Options"):
            st.session_state.input_max_workers = st.slider(
                "Maximum Parallel Workers", 1, 10,
                value=st.session_state.input_max_workers, # Read from state
                key='max_workers_slider',
                help="Higher values scrape pages faster but may trigger rate limits"
            )
            st.session_state.input_request_delay = st.slider(
                "Delay Between Requests (seconds)", 0.1, 5.0,
                value=st.session_state.input_request_delay, # Read from state
                key='request_delay_slider',
                help="Longer delays reduce risk of rate limiting"
            )
            
            # NEW CHECKBOX for Advanced Extraction Mode
            st.session_state.input_advanced_extraction = st.checkbox(
                "Use Advanced Extraction Mode",
                value=st.session_state.get('input_advanced_extraction', False),
                key='advanced_extraction_cb',
                help="Slower, but necessary for sites that render content with JavaScript. Use this if you get empty or minimal content from a site."
            )
            
            # NEW SLIDER for Content Block Aggregation
            st.session_state.input_num_blocks_to_combine = st.slider(
                "Number of Content Blocks to Combine",
                min_value=1,
                max_value=5,
                value=st.session_state.get('input_num_blocks_to_combine', 2),  # Safe and effective default
                key="num_blocks_slider",
                help="Defines how many of the largest text blocks from a page should be combined. Increase for fragmented pages (e.g., complex landing pages). Set to 1 for the most precise extraction on simple articles."
            )
            
            # Show warning if Advanced Mode is enabled but Chrome might not be available
            if st.session_state.get('input_advanced_extraction', False):
                try:
                    from modules.content_extractor import SELENIUM_AVAILABLE
                    if not SELENIUM_AVAILABLE:
                        st.warning("⚠️ Advanced Extraction Mode requires Selenium and Google Chrome. Please install them to use this feature.")
                except ImportError:
                    st.warning("⚠️ Advanced Extraction Mode requires additional dependencies. Please check your installation.")
        # Assign local variables AFTER widgets
        max_workers = st.session_state.input_max_workers
        request_delay = st.session_state.input_request_delay

    else: # If content analysis is off, show warning
        st.warning("⚠️ Content analysis is required for the new embedding system. Please enable it above.")
        max_workers = st.session_state.input_max_workers
        request_delay = st.session_state.input_request_delay


    # --- Advanced Analysis Options ---
    with st.expander("Advanced Analysis Options"):
        st.session_state.input_max_urls = st.slider(
            "Maximum URLs to analyze:", 10, 10000,
            value=st.session_state.input_max_urls, # Read from state
            key='max_urls_slider',
            help="Lower values are faster but less comprehensive"
        )
        st.session_state.input_perplexity = st.slider(
            "t-SNE Perplexity:", 5, 50,
            value=st.session_state.input_perplexity, # Read from state
            key='perplexity_slider',
            help="Lower values preserve local structure, higher values preserve global structure"
        )
        st.session_state.input_focus_k = st.slider(
            "Focus Score Scaling (k1):", 1.0, 20.0,
            value=st.session_state.input_focus_k, # Read from state
            key='focus_k_slider',
            help="Higher values make the focus score more sensitive to distance variations"
        )
        st.session_state.input_radius_k = st.slider(
            "Radius Score Scaling (k2):", 1.0, 20.0,
            value=st.session_state.input_radius_k, # Read from state
            key='radius_k_slider',
            help="Higher values make the radius score more sensitive to maximum distances"
        )
    # Assign local variables AFTER widgets
    max_urls = st.session_state.input_max_urls
    perplexity = st.session_state.input_perplexity
    focus_k = st.session_state.input_focus_k
    radius_k = st.session_state.input_radius_k

    # --- AI Summary Options ---
    st.session_state.input_use_ai_summary = st.checkbox(
        "Generate AI Summary with OpenRouter",
        value=st.session_state.input_use_ai_summary, # Read from state
        key='use_ai_summary_cb'
    )
    use_ai_summary = st.session_state.input_use_ai_summary # Assign local variable

    selected_model = None
    if use_ai_summary:
        # Get OpenRouter API key (from input or env)
        openrouter_key = st.session_state.input_openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if openrouter_key:
            # Fetch available models
            with st.spinner("Fetching available AI models..."):
                available_models = fetch_openrouter_models(openrouter_key)
            
            if available_models:
                model_options = [(f"{model['id']} ({model.get('name', 'Unknown')})", model['id']) for model in available_models]  # Show all available models
                
                if st.session_state.input_selected_model:
                    # Try to find the previously selected model
                    default_index = 0
                    for i, (_, model_id) in enumerate(model_options):
                        if model_id == st.session_state.input_selected_model:
                            default_index = i
                            break
                else:
                    default_index = 0
                
                selected_display = st.selectbox(
                    "Choose AI Model:",
                    options=[display for display, _ in model_options],
                    index=default_index,
                    key='model_select',
                    help="Select the AI model for generating summaries"
                )
                
                # Extract model ID from selection
                selected_model = next(model_id for display, model_id in model_options if display == selected_display)
                st.session_state.input_selected_model = selected_model
                
                st.success(f"✅ Selected model: {selected_model}")
            else:
                st.error("❌ Failed to fetch models from OpenRouter. Check your API key.")
        else:
            st.warning("⚠️ OpenRouter API key required for AI summaries. Please provide it above or in .env file.")    
    else:
        selected_model = None

    st.divider()

    # --- Save Analysis Section ---
    st.header("Save Analysis")
    # Suggest a filename based on the domain
    save_filename_default = f"{st.session_state.domain.replace('.', '_')}_analysis.pkl" if st.session_state.domain else "analysis.pkl"
    save_filename = st.text_input("Save analysis as:", value=save_filename_default, key="save_filename_input")

    # Only show save button if analysis is complete and not currently loading
    can_save = st.session_state.results_df is not None and not st.session_state._loading_in_progress
    if st.button("Save Analysis State", key="save_state_button", use_container_width=True, disabled=not can_save):
        if save_filename:
            save_analysis_state(save_filename)
        else:
            st.error("Please enter a filename to save the analysis.")
    elif not can_save and st.session_state.results_df is None:
        st.caption("Run an analysis first to enable saving.")

    st.divider()

    # --- Start Analysis Section ---
    st.header("Run New Analysis")
    
    # URL Input Section
    st.subheader("Enter URLs to Analyze")
    url_input_text = st.text_area(
        "Paste URLs (one per line):",
        height=150,
        help="Enter the URLs you want to analyze, one per line. They should be from the same domain.",
        key="url_input_area",
        placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3"
    )
    
    analyze_button = st.button("Start Analysis", use_container_width=True, key="start_analysis_button")


# --- Main Application Logic ---

# Helper function for logging to UI
def log_to_ui(message):
    """Add a message to the UI log"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(message)
    # Also print to console for debugging
    print(f"LOG: {message}")

def export_cannibalization_to_csv(displayed_duplicates):
    """Export cannibalization pairs to CSV with all required columns"""
    if not displayed_duplicates:
        st.warning("No cannibalization data to export.")
        return
    
    # Prepare export data
    export_data = []
    for pair in displayed_duplicates:
        url1 = pair['url1']
        url2 = pair['url2']
        distance = pair['distance']
        
        # Get content previews from results_df
        content_preview_1 = ""
        content_preview_2 = ""
        
        if 'content_preview' in st.session_state.results_df.columns:
            content1_match = st.session_state.results_df.loc[st.session_state.results_df['url'] == url1, 'content_preview']
            content2_match = st.session_state.results_df.loc[st.session_state.results_df['url'] == url2, 'content_preview']
            
            if len(content1_match) > 0:
                content_preview_1 = str(content1_match.values[0])
            if len(content2_match) > 0:
                content_preview_2 = str(content2_match.values[0])
        
        export_data.append({
            'URL1': url1,
            'URL2': url2,
            'Distance': distance,
            'Content_Preview_1': content_preview_1,
            'Content_Preview_2': content_preview_2
        })
    
    # Create DataFrame and CSV
    export_df = pd.DataFrame(export_data)
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cannibalization_analysis_{timestamp}.csv"
    
    # Offer download
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key="download_cannibalization_csv"
    )
    
    st.success(f"✅ CSV prepared with {len(export_data)} cannibalization pairs!")

# Local variables for analysis derived from session state (read AFTER widgets)
# These reflect the user's current selections for a *new* analysis run
# If analysis is loaded, these values might differ from the loaded state, which is fine.
domain = st.session_state.domain
include_filters = [f for f in st.session_state.input_include_filters if f] # Use non-empty filters
exclude_filters = [f for f in st.session_state.input_exclude_filters if f] # Use non-empty filters
include_logic_any = st.session_state.input_include_logic_any
analyze_content = st.session_state.input_analyze_content
max_workers = st.session_state.input_max_workers
request_delay = st.session_state.input_request_delay
max_urls = st.session_state.input_max_urls
perplexity = st.session_state.input_perplexity
focus_k = st.session_state.input_focus_k
radius_k = st.session_state.input_radius_k
use_ai_summary = st.session_state.input_use_ai_summary

# Get API keys (from input or environment)
jina_api_key = st.session_state.input_jina_api_key or os.getenv("JINA_API_KEY")
openrouter_api_key = st.session_state.input_openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
selected_model = st.session_state.input_selected_model

# --- Analysis Execution Block ---
# Only run if the "Start Analysis" button was clicked AND URLs are provided
# And crucially, only if an analysis wasn't just loaded
if analyze_button and url_input_text.strip():
    st.session_state.analysis_loaded = False # Explicitly mark as NOT loaded state
    st.session_state._last_uploaded_file = None # Clear last loaded file tracker

    # Initialize logging
    st.session_state.log_messages = []
    log_to_ui("🚀 Starting new analysis...")
    
    # Reset previous results before starting a new analysis
    keys_to_reset = ['sitemaps', 'selected_sitemaps', 'urls', 'url_sources', 'content_dict',
                     'results_df', 'focus_score', 'radius_score',
                     'pairwise_distances', 'llm_summary', 'centroid', 'content_for_embedding', 'embeddings_matrix']
    for key in keys_to_reset:
        st.session_state[key] = DEFAULT_SESSION_STATE.get(key) # Reset to default

    # Step 1: Process URL Input
    with st.spinner("Processing URLs..."):
        log_to_ui("📝 Parsing URL input...")
        
        # Parse URLs from text input
        input_urls = []
        lines = url_input_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('http://') or line.startswith('https://')):
                input_urls.append(line)
        
        log_to_ui(f"🔍 Found {len(input_urls)} URLs in input")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in input_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        if len(unique_urls) != len(input_urls):
            log_to_ui(f"🔄 Removed {len(input_urls) - len(unique_urls)} duplicate URLs")
        
        # Validate domain consistency if domain is provided
        if domain:
            from urllib.parse import urlparse
            domain_to_check = domain.lower().replace('www.', '')
            filtered_urls = []
            skipped_count = 0
            for url in unique_urls:
                parsed = urlparse(url)
                url_domain = parsed.netloc.lower().replace('www.', '')
                if domain_to_check in url_domain or url_domain in domain_to_check:
                    filtered_urls.append(url)
                else:
                    st.warning(f"Skipping URL from different domain: {url}")
                    skipped_count += 1
            unique_urls = filtered_urls
            if skipped_count > 0:
                log_to_ui(f"⚠️ Skipped {skipped_count} URLs from different domains")
        
        # Store URLs directly - no sitemap needed
        st.session_state.urls = unique_urls
        st.session_state.url_sources = {url: "Manual Input" for url in unique_urls}
        st.session_state.sitemaps = ["Manual Input"]  # For compatibility
        st.session_state.selected_sitemaps = ["Manual Input"]  # For compatibility
        
        log_to_ui(f"✅ Processed {len(unique_urls)} unique URLs ready for analysis")

    if not unique_urls:
        st.error("No valid URLs found. Please check your input and ensure URLs start with http:// or https://")
    else:
        st.success(f"Found {len(unique_urls)} unique URLs!")
        
        # Show preview of URLs
        with st.expander("URLs to be analyzed (click to expand)"):
            for i, url in enumerate(unique_urls[:20]):
                st.write(f"{i+1}. {url}")
            if len(unique_urls) > 20:
                st.write(f"... and {len(unique_urls) - 20} more")
        
        st.rerun() # Rerun to show URL list and start processing

# Step 2: Show URL List (Show if URLs are available OR if loaded state has URLs)
if st.session_state.urls:
    st.subheader("URLs Ready for Analysis")
    url_cols = st.columns(2)
    process_button_disabled = False  # URLs are already processed and ready

    with url_cols[0]:
        # If analysis was loaded, just display info about loaded URLs
        if st.session_state.analysis_loaded:
            st.markdown("**URLs (from loaded analysis):**")
            st.write(f"Total URLs: {len(st.session_state.urls)}")
        # Otherwise (new analysis), show URL info
        else:
            st.markdown("**URLs from manual input:**")
            st.write(f"Total URLs: {len(st.session_state.urls)}")
            
            # Apply filters before processing
            filtered_count = 0
            if st.session_state.urls:
                def url_passes_filters_local(url, inc_filters, exc_filters, inc_logic_any):
                    for exclude in exc_filters:
                        if exclude and exclude.lower() in url.lower(): return False
                    if not inc_filters: return True # Pass if no include filters
                    if inc_logic_any:
                        return any(include.lower() in url.lower() for include in inc_filters if include)
                    else:
                        return all(include.lower() in url.lower() for include in inc_filters if include)
                
                for url in st.session_state.urls:
                    if url_passes_filters_local(url, include_filters, exclude_filters, include_logic_any):
                        filtered_count += 1
                
                if filtered_count != len(st.session_state.urls):
                    st.info(f"After applying filters: {filtered_count} URLs will be processed")

    with url_cols[1]:
        # Show URL count and source info
        st.markdown(f"**Source: Manual Input**")
        if st.session_state.urls:
            with st.expander("Preview URLs (first 10)"):
                for i, url in enumerate(st.session_state.urls[:10]):
                    st.write(f"{i+1}. {url}")
                if len(st.session_state.urls) > 10:
                    st.write(f"... and {len(st.session_state.urls) - 10} more")

    # "Process URLs" Button - only shown if NOT loaded
    if not st.session_state.analysis_loaded:
        process_button = st.button(
            "Process URLs",
            use_container_width=True,
            disabled=process_button_disabled,
            key="process_urls_button"
        )

        # Step 3: Process URLs (Run only if Process button clicked AND URLs available AND not loaded)
        if process_button and st.session_state.urls:
            # --- Start of Processing Block (Modified for Direct URL Input) ---
            with st.spinner("Applying filters to URLs..."):
                all_urls_list = st.session_state.urls[:] # Start with input URLs
                url_sources_dict = st.session_state.url_sources.copy() # Already set to "Manual Input"

                # Define filter function locally
                def url_passes_filters_local(url, inc_filters, exc_filters, inc_logic_any):
                     for exclude in exc_filters:
                         if exclude and exclude.lower() in url.lower(): return False
                     if not inc_filters: return True # Pass if no include filters
                     if inc_logic_any:
                         return any(include.lower() in url.lower() for include in inc_filters if include)
                     else:
                         return all(include.lower() in url.lower() for include in inc_filters if include)

                # Apply filters to input URLs
                st.info(f"Applying filters to {len(all_urls_list)} input URLs...")
                filtered_urls = [
                    url for url in all_urls_list
                    if url_passes_filters_local(url, include_filters, exclude_filters, include_logic_any)
                ]
                
                # Update URL sources for filtered URLs only
                url_sources_dict = {url: "Manual Input" for url in filtered_urls}
                
                if len(filtered_urls) != len(all_urls_list):
                    st.info(f"Filters removed {len(all_urls_list) - len(filtered_urls)} URLs")
                
                unique_urls_list = filtered_urls # Already unique from input processing

                if len(unique_urls_list) > max_urls: # Use current max_urls
                    log_to_ui(f"⚠️ Limiting analysis to {max_urls} URLs out of {len(unique_urls_list)} filtered")
                    st.warning(f"Limiting analysis to {max_urls} URLs out of {len(unique_urls_list)} filtered.")
                    unique_urls_list = unique_urls_list[:max_urls]
                    url_sources_dict = {url: source for url, source in url_sources_dict.items() if url in unique_urls_list}

                # Update session state with filtered results
                st.session_state.urls = unique_urls_list
                st.session_state.url_sources = url_sources_dict

                # Display sample (Modified Logic)
                if unique_urls_list:
                    with st.expander("URLs to be analyzed (click to expand)"):
                         for i, url in enumerate(unique_urls_list[:10]):
                             source = url_sources_dict.get(url, "Unknown")
                             st.write(f"{i+1}. {url} (from: {source})")
                         if len(unique_urls_list) > 10: st.write(f"... and {len(unique_urls_list) - 10} more")
                time.sleep(0.5) # Original delay

            if not st.session_state.urls:
                st.error("No URLs remain after applying filters. Please adjust your filter criteria.")
            else:
                st.success(f"Ready to analyze {len(st.session_state.urls)} URLs!")

                # --- Content Extraction Step (Original Logic Preserved) ---
                if analyze_content: # Use current setting
                    with st.spinner("Extracting page content... This may take a while..."):
                        st.info(f"Extracting content from {len(st.session_state.urls)} pages with {max_workers} parallel workers...") # Use current settings
                        st.warning("This step may take several minutes...")

                        progress_bar = st.progress(0)
                        status_text = st.empty() # Placeholder for text updates

                        # --- Original Function: extract_with_progress ---
                        def extract_with_progress(urls_to_scrape):
                            results = {}
                            total = len(urls_to_scrape)
                            for i, url in enumerate(urls_to_scrape):
                                try:
                                    # CORE LOGIC CHANGE: Choose the extraction function based on advanced mode setting
                                    if st.session_state.get('input_advanced_extraction', False):
                                        from modules.content_extractor import _extract_content_with_headless_browser
                                        content = _extract_content_with_headless_browser(url, num_blocks_to_combine=st.session_state.get('input_num_blocks_to_combine', 2))
                                    else:
                                        content = extract_main_content(url, num_blocks_to_combine=st.session_state.get('input_num_blocks_to_combine', 2))
                                    results[url] = content

                                    progress = (i + 1) / total
                                    progress_bar.progress(progress)
                                    if (i + 1) % 5 == 0 or (i + 1) == total:
                                         status_text.text(f"Processed {i + 1} of {total} URLs...") # Update status text

                                    if i < total - 1: time.sleep(request_delay) # Use current delay

                                except Exception as e:
                                    st.error(f"Error extracting content from {url}: {str(e)}")
                                    results[url] = "" # Store empty string on error
                            status_text.text("Sequential extraction complete.")
                            return results
                        # --- End Original Function ---

                        # --- Original Conditional Extraction Logic ---
                        if len(st.session_state.urls) <= 20:
                            status_text.text("Using sequential extraction for small site...")
                            content_dict_result = extract_with_progress(st.session_state.urls)
                        else:
                            status_text.text("Using parallel extraction...")
                            # Assuming batch_extract_content handles progress internally or add callback if supported
                            content_dict_result = batch_extract_content(
                                st.session_state.urls,
                                max_workers=max_workers, # Use current setting
                                delay=request_delay, # Use current setting
                                use_advanced_extraction=st.session_state.get('input_advanced_extraction', False), # PASS THE CHECKBOX STATE
                                num_blocks_to_combine=st.session_state.get('input_num_blocks_to_combine', 2) # PASS THE SLIDER VALUE
                            )
                            progress_bar.progress(1.0) # Ensure completion
                            status_text.text("Parallel extraction complete.")
                        # --- End Original Conditional Logic ---

                        st.session_state.content_dict = content_dict_result


                        # Show stats and preview (Original Logic)
                        content_lengths = [len(str(c)) for c in content_dict_result.values()]
                        avg_len = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                        empty_count = sum(1 for length in content_lengths if length == 0)
                        st.info(f"Content extraction complete! Average length: {avg_len:.1f} chars")
                        if empty_count > 0: st.warning(f"Could not extract content from {empty_count} URLs")

                        with st.expander("Sample of extracted content (click to expand)"):
                            for i, (url, content) in enumerate(list(content_dict_result.items())[:3]):
                                source = st.session_state.url_sources.get(url, "Unknown")
                                st.write(f"**URL:** {url} (from: {source})")
                                preview = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                                st.text_area(f"Content preview {i+1}", preview, height=150, key=f"extract_preview_{i}")
                        time.sleep(0.5) # Original delay
                else: # If not analyzing content
                    st.session_state.content_dict = None


                # --- Vectorization Step (New Jina Embeddings) ---
                with st.spinner("Generating embeddings with Jina AI..."):
                    log_to_ui("🔄 Starting Jina embedding generation...")
                    
                    if not jina_api_key:
                        st.error("❌ Jina API key is required for embedding generation. Please provide it in the sidebar or .env file.")
                        log_to_ui("❌ Missing Jina API key")
                        st.stop()
                    
                    # Use raw content_dict for embedding generation (not processed!)
                    content_input = st.session_state.content_dict if analyze_content else None
                    
                    if not content_input:
                        st.error("❌ No content available for embedding generation. Content analysis must be enabled.")
                        log_to_ui("❌ No content available for embeddings")
                        st.stop()
                    
                    # Prepare raw content list in same order as URLs
                    raw_content_list_for_embedding = []
                    for url in st.session_state.urls:
                        if url in content_input:
                            raw_content_list_for_embedding.append(content_input[url])
                        else:
                            raw_content_list_for_embedding.append("")
                    
                    # Generate embeddings using new Jina system
                    matrix_res = generate_jina_embeddings(
                        raw_content_list_for_embedding, 
                        jina_api_key, 
                        log_to_ui
                    )
                    
                    if matrix_res is None:
                        st.error("❌ Failed to generate embeddings. Check your API key and content.")
                        log_to_ui("❌ Embedding generation failed")
                        st.stop()
                    
                    # Store embeddings and raw content for embedding
                    st.session_state.embeddings_matrix = matrix_res
                    st.session_state.content_for_embedding = raw_content_list_for_embedding
                    
                    log_to_ui(f"✅ Generated embeddings matrix: {matrix_res.shape[0]} URLs × {matrix_res.shape[1]} dimensions")
                    
                    # Store embeddings matrix for interactive clustering (Phase 2)
                    st.session_state.embeddings_matrix = matrix_res
                    log_to_ui("💾 Embeddings matrix saved for interactive clustering")
                    
                    # Prepare return values for compatibility
                    url_list_res = st.session_state.urls
                    processed_paths_res = [""] * len(st.session_state.urls)  # No URL path processing in new version
                    vectorizer_res = None  # No vectorizer object in new system

                    with st.expander("Vectorization Details (click to expand)"):
                         # Explanations (Updated for Jina embeddings)
                         st.subheader("How Vectorization Works")
                         if analyze_content: 
                             st.markdown("**Jina Embeddings v4 Content Analysis:**")
                             st.markdown("This analysis uses advanced semantic embeddings from Jina AI to understand the meaning and context of your page content. Unlike traditional TF-IDF analysis, these embeddings capture semantic relationships between concepts, providing more accurate topical analysis.")
                         else: 
                             st.markdown("**No Analysis Mode:**")
                             st.markdown("Content analysis is disabled. Please enable it to use the new Jina embedding system.")
                         # Samples (Keep original logic)
                         st.subheader("Sample Data")
                         for i in range(min(3, len(url_list_res))):
                              url = url_list_res[i]
                              source = st.session_state.url_sources.get(url, "Unknown")
                              st.write(f"**URL:** {url} (from: {source})")
                              st.write(f"**Processed Path:** {processed_paths_res[i]}")
                              if analyze_content and i < len(raw_content_list_for_embedding):
                                   preview = raw_content_list_for_embedding[i][:200] + "..."
                                   st.write(f"**Processed Content:** {preview}")
                              st.write("---")
                    time.sleep(0.5) # Original delay


                # --- Dimensionality Reduction (Original Logic Preserved) ---
                with st.spinner("Reducing dimensions (t-SNE)... This may take time..."):
                    st.info("Starting t-SNE dimensionality reduction...")
                    st.warning("This step may take several minutes for larger datasets!")
                    progress_placeholder = st.empty()
                    progress_placeholder.text("Running t-SNE...")

                    # Ensure perplexity is valid based on current setting and matrix size
                    num_samples = matrix_res.shape[0]
                    safe_perplexity = min(perplexity, num_samples - 1) if num_samples > 1 else 1
                    if safe_perplexity != perplexity and num_samples > 1:
                         st.warning(f"Perplexity adjusted from {perplexity} to {safe_perplexity} due to dataset size ({num_samples}).")

                    coordinates_df_res, centroid_res = reduce_dimensions_and_find_centroid(
                        matrix_res,
                        perplexity=safe_perplexity
                    )
                    st.session_state.centroid = centroid_res # Store result

                    progress_placeholder.empty() # Clear progress text
                    st.info(f"t-SNE complete. Centroid: ({centroid_res[0]:.2f}, {centroid_res[1]:.2f})")
                    time.sleep(0.5) # Original delay


                # --- Calculate Metrics (Original Logic Preserved) ---
                with st.spinner("Calculating metrics..."):
                    # Pass current k values from sliders and embedding matrix for semantic analysis
                    results_df_res, focus_score_res, radius_score_res, pairwise_dist_matrix_res = calculate_metrics(
                        url_list=url_list_res,
                        processed_paths=processed_paths_res,
                        coordinates_df=coordinates_df_res,
                        centroid=centroid_res,
                        k1=focus_k, # Current slider value
                        k2=radius_k, # Current slider value
                        embedding_matrix=matrix_res  # Original embeddings for precise semantic analysis
                    )

                    # Add content preview (Original logic, uses processed_content list from state)
                    if analyze_content and st.session_state.content_for_embedding and len(st.session_state.content_for_embedding) == len(results_df_res):
                        results_df_res['content_preview'] = [
                            (p[:200] + "..." if len(p) > 200 else p)
                            for p in st.session_state.content_for_embedding
                        ]

                    # Add source sitemap (Original Logic)
                    results_df_res['source_sitemap'] = results_df_res['url'].apply(
                        lambda url: st.session_state.url_sources.get(url, "Unknown")
                    )

                    # Store results in session state
                    st.session_state.results_df = results_df_res
                    st.session_state.focus_score = focus_score_res
                    st.session_state.radius_score = radius_score_res
                    st.session_state.pairwise_distances = pairwise_dist_matrix_res
                    
                    # Phase 1 complete - clustering will be done interactively in Phase 2

                    st.info(f"Metrics calculated. Focus Score: {focus_score_res:.1f}, Radius Score: {radius_score_res:.1f}")


                # --- Generate AI Summary (New OpenRouter System) ---
                if use_ai_summary and selected_model: # Use current setting
                    with st.spinner("Generating AI summary..."):
                        log_to_ui("🤖 Starting AI summary generation...")
                        
                        if not openrouter_api_key:
                            st.error("OpenRouter API key is required for AI summaries.")
                            log_to_ui("❌ Missing OpenRouter API key")
                            st.session_state.llm_summary = "Error: OpenRouter API key not provided."
                        else:
                            # Prepare data for summary
                            sorted_df = st.session_state.results_df.sort_values('distance_from_centroid')
                            top_focused = sorted_df['url'].head(5).tolist()
                            top_divergent = sorted_df['url'].tail(5).tolist()
                            page_types = sorted_df['page_type'].value_counts().to_dict() if 'page_type' in sorted_df.columns else None

                            # Generate summary using OpenRouter
                            summary = generate_summary_with_openrouter(
                                api_key=openrouter_api_key,
                                model_id=selected_model,
                                focus_score=st.session_state.focus_score,
                                radius_score=st.session_state.radius_score,
                                total_urls=len(url_list_res),
                                top_focused_urls=top_focused,
                                top_divergent_urls=top_divergent,
                                page_type_distribution=page_types,
                                logger_callback=log_to_ui
                            )
                            
                            if summary:
                                st.session_state.llm_summary = summary
                                log_to_ui("✅ AI summary generated successfully")
                            else:
                                st.session_state.llm_summary = "Error: Failed to generate AI summary."
                                log_to_ui("❌ AI summary generation failed")
                elif use_ai_summary and not selected_model:
                    log_to_ui("⚠️ AI summary requested but no model selected")
                    st.session_state.llm_summary = "Error: No AI model selected for summary generation."
            # --- End of Processing Block ---
            log_to_ui("🎉 Analysis processing complete!")
            st.success("Analysis processing complete!")
            st.rerun() # Rerun to display results immediately

elif analyze_button and not url_input_text.strip():
    st.warning("Please enter URLs in the text area before starting analysis.")
elif analyze_button and not analyze_content:
    st.warning("Please enable content analysis above - it's required for the new embedding system.")

# --- Display Results ---
# This section runs if results_df exists (from new run OR loaded state)
if st.session_state.results_df is not None:
    if not st.session_state.analysis_loaded :
         st.success("Displaying new analysis results.")
    else:
         st.success("Displaying loaded analysis results.")
    
    # Show detailed log if available
    if st.session_state.log_messages:
        with st.expander("📋 Detailed Processing Log", expanded=False):
            for i, message in enumerate(st.session_state.log_messages):
                st.text(f"{i+1:2d}. {message}")

    # Determine tabs based on available data
    tab_titles = ["Overview", "URL Details", "Visual Map (t-SNE)", "Cannibalization/Clusters"]
    content_available = st.session_state.content_dict is not None
    embeddings_available = st.session_state.embeddings_matrix is not None
    
    # Topic Cluster Analysis tab is available when we have embeddings from Phase 1
    if embeddings_available:
        tab_titles.append("Topic Cluster Analysis")
    # Internal Semantic Search tab is available when we have content data
    if content_available:
        tab_titles.append("Internal Semantic Search")
    if content_available:
         tab_titles.append("Content Inspector")

    tabs = st.tabs(tab_titles)

    # Tab 1: Overview (Keep original content)
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1: st.metric("Site Focus Score", f"{st.session_state.focus_score:.1f}/100", help="...")
        with col2: st.metric("Site Radius Score", f"{st.session_state.radius_score:.1f}/100", help="...")

        # Sitemap Distribution (Original logic)
        unique_sources = set(st.session_state.url_sources.values())
        if len(unique_sources) > 1 and 'source_sitemap' in st.session_state.results_df.columns:
            st.subheader("Sitemap Distribution")
            sitemap_counts = st.session_state.results_df['source_sitemap'].value_counts()
            fig_sitemap = px.pie(values=sitemap_counts.values, names=sitemap_counts.index, title="URLs by Source Sitemap")
            st.plotly_chart(fig_sitemap, use_container_width=True)

        # LLM Summary (Original logic)
        if st.session_state.llm_summary:
            st.subheader("Analysis")
            st.markdown(st.session_state.llm_summary)

        # Page Type Distribution (Original logic)
        if 'page_type' in st.session_state.results_df.columns:
             st.subheader("Page Type Distribution")
             page_type_counts = st.session_state.results_df['page_type'].value_counts()
             if not page_type_counts.empty:
                 fig_pie = px.pie(values=page_type_counts.values, names=page_type_counts.index, title="Content Distribution by Page Type")
                 st.plotly_chart(fig_pie, use_container_width=True)
             else:
                 st.info("No page type information available.")


        # Distance Distribution (Original logic)
        if 'distance_from_centroid' in st.session_state.results_df.columns:
             st.subheader("Distance Distribution")
             fig_hist = px.histogram(st.session_state.results_df, x="distance_from_centroid", nbins=30, title="Distribution of URL Distances", labels={"distance_from_centroid": "Distance"})
             st.plotly_chart(fig_hist, use_container_width=True)

        # Focused/Divergent URLs (Original logic)
        col1, col2 = st.columns(2)
        if 'distance_from_centroid' in st.session_state.results_df.columns:
             with col1:
                  st.subheader("Most Focused URLs")
                  focused_df = st.session_state.results_df.nsmallest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']].rename(columns={'distance_from_centroid': 'distance'})
                  st.dataframe(focused_df, hide_index=True, use_container_width=True)
             with col2:
                  st.subheader("Most Divergent URLs")
                  divergent_df = st.session_state.results_df.nlargest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']].rename(columns={'distance_from_centroid': 'distance'})
                  st.dataframe(divergent_df, hide_index=True, use_container_width=True)

    # Tab 2: URL Details (Enhanced with export functionality)
    with tabs[1]:
        st.subheader("URL Analysis Details")
        # Filters (Original logic)
        filter_cols = st.columns([2, 1, 1])
        with filter_cols[0]: search_term = st.text_input("Search URLs:", key="details_search")
        with filter_cols[1]:
            # Check if 'page_type' column exists before creating filter options
            all_page_types = ["All"]
            if 'page_type' in st.session_state.results_df.columns:
                all_page_types.extend(sorted(st.session_state.results_df['page_type'].unique().tolist()))
            selected_page_type = st.selectbox("Filter by Page Type:", all_page_types, key="details_type_filter")
        with filter_cols[2]:
             # Check if 'source_sitemap' column exists
            all_sitemaps = ["All"]
            sitemap_filter_disabled = True
            if 'source_sitemap' in st.session_state.results_df.columns:
                 unique_sitemaps = sorted(st.session_state.results_df['source_sitemap'].unique().tolist())
                 all_sitemaps.extend(unique_sitemaps)
                 sitemap_filter_disabled = len(unique_sitemaps) <= 1
            selected_sitemap = st.selectbox("Filter by Sitemap:", all_sitemaps, key="details_sitemap_filter", disabled=sitemap_filter_disabled)

        # Apply Filters (Original logic - ensure columns exist before filtering)
        filtered_df = st.session_state.results_df.copy()
        if search_term and 'url' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['url'].str.contains(search_term, case=False)]
        if selected_page_type != "All" and 'page_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['page_type'] == selected_page_type]
        if selected_sitemap != "All" and 'source_sitemap' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['source_sitemap'] == selected_sitemap]

        # --- MODIFICATION START ---
        # Prepare for display: Add 'x' and 'y' to the list of desired columns
        display_columns = ['url', 'page_type', 'source_sitemap', 'page_depth', 'distance_from_centroid', 'x', 'y']
        if 'content_preview' in filtered_df.columns:
             display_columns.append('content_preview') # Add preview if it exists

        # Ensure all desired display columns actually exist in the filtered dataframe before selection
        existing_display_columns = [col for col in display_columns if col in filtered_df.columns]
        # --- MODIFICATION END ---

        if existing_display_columns:
             display_df = filtered_df[existing_display_columns] # Select only existing columns

             # Rename columns for display
             rename_map = {
                 'distance_from_centroid': 'distance',
                 'page_depth': 'depth',
                 'source_sitemap': 'sitemap'
             }
             # Only rename columns that actually exist in display_df
             actual_rename_map = {k: v for k, v in rename_map.items() if k in display_df.columns}
             display_df = display_df.rename(columns=actual_rename_map)

             # --- MODIFICATION START ---
             # Update column config to include 'x' and 'y'
             column_config_dict = {
                 "url": st.column_config.TextColumn("URL", help="The URL analyzed"),
                 "page_type": st.column_config.TextColumn("Page Type", help="Categorization based on URL structure"),
                 "sitemap": st.column_config.TextColumn("Sitemap", help="Source sitemap the URL was found in"),
                 "depth": st.column_config.NumberColumn("Depth", help="URL path depth"),
                 "distance": st.column_config.NumberColumn("Distance", help="Euclidean distance from the topic centroid in the t-SNE map", format="%.3f"),
                 "x": st.column_config.NumberColumn("Coord X", help="X-coordinate from the t-SNE visualization", format="%.3f"),
                 "y": st.column_config.NumberColumn("Coord Y", help="Y-coordinate from the t-SNE visualization", format="%.3f"),
                 "content_preview": st.column_config.TextColumn("Content Preview", help="First 200 characters of processed content (if available)"),
             }

             # Filter column config to only include columns present in the final display_df
             final_column_config = {col: config for col, config in column_config_dict.items() if col in display_df.columns}
             # --- MODIFICATION END ---

             st.dataframe(
                 # Sort by distance if the column exists after potential renaming
                 display_df.sort_values('distance') if 'distance' in display_df.columns else display_df,
                 use_container_width=True,
                 column_config=final_column_config # Use the filtered config
             )
             st.info(f"Showing {len(filtered_df)} URLs out of {len(st.session_state.results_df)} total.")
             
             # Export functionality
             st.subheader("📊 Export Options")
             export_cols = st.columns(2)
             
             with export_cols[0]:
                 # Standard CSV export
                 if st.button("📄 Export URL Data to CSV", use_container_width=True):
                     csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                     st.download_button(
                         label="Download CSV",
                         data=csv_data,
                         file_name=f"url_analysis_{domain or 'data'}.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
             
             with export_cols[1]:
                 # Embeddings export
                 if st.session_state.embeddings_matrix is not None:
                     if st.button("🧪 Export Embeddings to CSV", use_container_width=True):
                         # Create embeddings export DataFrame
                         embeddings_export_data = []
                         
                         for i, url in enumerate(st.session_state.urls):
                             if i < len(st.session_state.embeddings_matrix):
                                 # Convert embedding vector to string representation
                                 embedding_vector = st.session_state.embeddings_matrix[i]
                                 embedding_str = ','.join(map(str, embedding_vector))
                                 
                                 # Get additional data
                                 source = st.session_state.url_sources.get(url, "Unknown")
                                 
                                 # Get distance if available
                                 distance = None
                                 if st.session_state.results_df is not None and url in st.session_state.results_df['url'].values:
                                     url_row = st.session_state.results_df[st.session_state.results_df['url'] == url].iloc[0]
                                     distance = url_row.get('distance_from_centroid')
                                 
                                 embeddings_export_data.append({
                                     'url': url,
                                     'source': source,
                                     'distance_from_centroid': distance,
                                     'embedding_dimensions': len(embedding_vector),
                                     'embedding_vector': embedding_str
                                 })
                         
                         if embeddings_export_data:
                             embeddings_df = pd.DataFrame(embeddings_export_data)
                             embeddings_csv = embeddings_df.to_csv(index=False).encode('utf-8')
                             
                             st.download_button(
                                 label="Download Embeddings CSV",
                                 data=embeddings_csv,
                                 file_name=f"embeddings_{domain or 'data'}.csv",
                                 mime="text/csv",
                                 use_container_width=True
                             )
                             
                             st.success(f"✅ Ready to export {len(embeddings_export_data)} embeddings")
                         else:
                             st.error("❌ No embedding data to export")
                 else:
                     st.info("📊 No embeddings available for export. Run a new analysis first.")
        else:
             st.warning("No columns available for display based on current data or filters.")


    # Tab 3: Visual Map (t-SNE) (Keep original content)
    with tabs[2]:  # Visual Map is always third
        st.subheader("Topical Map Visualization")
        # Options (Original logic)
        visual_cols = st.columns([2, 1, 1])
        with visual_cols[0]:
            color_options = {"Distance from Centroid": "distance_from_centroid", "Page Type": "page_type", "Page Depth": "page_depth", "Source Sitemap": "source_sitemap", "Topic Cluster": "Topic Cluster"}
            # Filter available color options based on columns present in results_df
            available_color_options = {k: v for k, v in color_options.items() if v in st.session_state.results_df.columns}
            if not available_color_options:
                st.warning("No data columns available for coloring points.")
                color_by_selection = None
            else:
                # Default to Topic Cluster if available
                default_color = "Topic Cluster" if "Topic Cluster" in available_color_options else list(available_color_options.keys())[0]
                color_by_selection = st.selectbox("Color points by:", list(available_color_options.keys()), 
                                                index=list(available_color_options.keys()).index(default_color), key="viz_color_select")

        with visual_cols[1]: point_size = st.slider("Point Size:", 3, 15, 8, key="viz_size_slider")
        with visual_cols[2]:
             all_sitemaps_viz = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist()) if 'source_sitemap' in st.session_state.results_df.columns else ["All"]
             viz_sitemap = st.selectbox("Show sitemap:", all_sitemaps_viz, key="viz_sitemap_filter", disabled=len(all_sitemaps_viz) <= 1)

        # Filter data (Original logic)
        viz_df = st.session_state.results_df.copy()
        if viz_sitemap != "All" and 'source_sitemap' in viz_df.columns: viz_df = viz_df[viz_df['source_sitemap'] == viz_sitemap]

        # Create plot if possible (Original logic + checks)
        if color_by_selection and 'x' in viz_df.columns and 'y' in viz_df.columns:
            color_column = available_color_options[color_by_selection]
            hover_cols = [col for col in ["page_type", "distance_from_centroid", "page_depth", "source_sitemap"] if col in viz_df.columns]

            if color_column in ["page_type", "source_sitemap", "Topic Cluster"]: # Categorical coloring
                 fig = px.scatter(viz_df, x="x", y="y", color=color_column, hover_name="url", hover_data=hover_cols, title="t-SNE Clustering", size_max=point_size)
            else: # Continuous coloring
                 fig = px.scatter(viz_df, x="x", y="y", color=color_column, color_continuous_scale="Viridis", hover_name="url", hover_data=hover_cols, title="t-SNE Clustering", size_max=point_size)

            # Add centroid (Original logic)
            if st.session_state.centroid is not None:
                 fig.add_trace(go.Scatter(x=[st.session_state.centroid[0]], y=[st.session_state.centroid[1]], mode="markers", marker=dict(symbol="star", size=15, color="red"), name="Centroid"))

            fig.update_layout(height=700, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **How to interpret this visualization:**
            * Each point represents a URL from the sitemap
            * Points that cluster together have similar topics
            * The star marker represents the "topic centroid" - the center of all topics
            * Distances from the centroid reflect how focused or divergent each URL is
            * Colors help identify patterns in content structure
            """) # Keep original info text
        elif 'x' not in viz_df.columns or 'y' not in viz_df.columns:
            st.warning("t-SNE coordinates (x, y) not found in results. Cannot generate map.")
        else:
            st.warning("Select a valid coloring option.")


    # Tab 4: Cannibalization/Clusters (Keep original content)
    with tabs[3]:  # Cannibalization is always fourth
        st.subheader("Content Cannibalization Analysis")
        st.markdown("This tab helps identify potentially duplicate content...")
        # Filters (Original logic)
        cann_cols = st.columns([2, 1, 1])
        with cann_cols[0]: threshold = st.slider("Distance Threshold:", min_value=0.01, max_value=2.0, value=0.5, step=0.01, help="...", key="cann_threshold") # Adjusted min/step potentially
        with cann_cols[1]: max_pairs = st.number_input("Max pairs to display:", 5, 100, 20, key="cann_max_pairs")
        with cann_cols[2]:
             all_sitemaps_cann = ["All pairs", "Same sitemap", "Different sitemaps"]
             cann_filter = st.selectbox("Filter pairs:", all_sitemaps_cann, key="cann_sitemap_filter", disabled=len(st.session_state.results_df['source_sitemap'].unique()) <= 1 if 'source_sitemap' in st.session_state.results_df.columns else True)

        # Analysis (Original logic + Preserved Expander)
        if st.session_state.results_df is not None and st.session_state.pairwise_distances is not None:

             # --- Start: Preserved Distance Matrix Statistics ---
             with st.expander("Distance Matrix Statistics"):
                 dist_matrix = st.session_state.pairwise_distances
                 st.write(f"Matrix shape: {dist_matrix.shape}")
                 # Calculate stats safely for potentially non-square matrices or if empty
                 if dist_matrix.size > 0:
                      non_zero_distances = dist_matrix[dist_matrix > 1e-6] # Avoid floating point zeros
                      st.write(f"Min non-zero distance: {non_zero_distances.min():.4f}" if non_zero_distances.size > 0 else "N/A")
                      st.write(f"Max distance: {dist_matrix.max():.4f}")
                      st.write(f"Mean non-zero distance: {non_zero_distances.mean():.4f}" if non_zero_distances.size > 0 else "N/A")
                      st.write(f"Number of close-to-zero distances: {(dist_matrix <= 1e-6).sum()}")

                      # Histogram of non-zero distances
                      if non_zero_distances.size > 0:
                          fig_dist_hist = px.histogram(
                              x=non_zero_distances.flatten(),
                              nbins=50,
                              title="Distribution of Pairwise Distances (excluding self-comparisons)",
                              labels={"x": "Distance"}
                          )
                          st.plotly_chart(fig_dist_hist)
                      else:
                           st.write("No non-zero distances to plot.")
                 else:
                      st.write("Distance matrix is empty.")
            # --- End: Preserved Distance Matrix Statistics ---

             duplicates = find_potential_duplicates(
                 st.session_state.results_df,
                 st.session_state.pairwise_distances,
                 threshold
             )

             # Filter duplicates (Original logic, check column exists)
             if cann_filter != "All pairs" and duplicates and 'source_sitemap' in st.session_state.results_df.columns:
                filtered_duplicates = []
                # Need url_sources mapping for this filter
                url_to_source = pd.Series(st.session_state.results_df.source_sitemap.values, index=st.session_state.results_df.url).to_dict()
                for dup in duplicates:
                     url1, url2 = dup['url1'], dup['url2']
                     source1 = url_to_source.get(url1, "Unknown")
                     source2 = url_to_source.get(url2, "Unknown")
                     if cann_filter == "Same sitemap" and source1 == source2: filtered_duplicates.append(dup)
                     elif cann_filter == "Different sitemaps" and source1 != source2: filtered_duplicates.append(dup)
                duplicates = filtered_duplicates

             # Display (Original logic)
             displayed_duplicates = duplicates[:max_pairs] if duplicates else []
             if not displayed_duplicates: st.info("No potential duplicates found with current settings.")
             else:
                 total_count = len(duplicates)
                 col1, col2 = st.columns([3, 1])
                 with col1:
                     st.success(f"Found {total_count} potential duplicates! Showing top {len(displayed_duplicates)}.")
                 with col2:
                     # CSV Export button
                     if st.button("📊 Export to CSV", key="export_cannibalization_csv", use_container_width=True):
                         export_cannibalization_to_csv(displayed_duplicates)
                 for i, row in enumerate(displayed_duplicates):
                      url1, url2 = row['url1'], row['url2']
                      # Need url_sources mapping again
                      url_to_source = pd.Series(st.session_state.results_df.source_sitemap.values, index=st.session_state.results_df.url).to_dict()
                      source1 = url_to_source.get(url1, "Unknown")
                      source2 = url_to_source.get(url2, "Unknown")

                      title = f"Pair {i+1}: Distance {row['distance']:.3f}"
                      if source1 != source2: title += f" (Cross-Sitemap: {source1} -> {source2})"
                      with st.expander(title):
                           # Display pair details (Original layout)
                           cols=st.columns(2)
                           with cols[0]:
                                st.markdown(f"**URL 1:** [{url1}]({url1})")
                                if 'processed_path' in row: st.text(f"Processed path: {row['path1']}") # Check if path info present
                                st.text(f"Source: {source1}")
                                if 'content_preview' in st.session_state.results_df.columns:
                                     content1 = st.session_state.results_df.loc[st.session_state.results_df['url'] == url1, 'content_preview'].values
                                     if len(content1)>0: st.text_area("Content preview 1:", content1[0], height=150, key=f"cann_p1_{i}")
                           with cols[1]:
                                st.markdown(f"**URL 2:** [{url2}]({url2})")
                                if 'processed_path' in row: st.text(f"Processed path: {row['path2']}") # Check if path info present
                                st.text(f"Source: {source2}")
                                if 'content_preview' in st.session_state.results_df.columns:
                                     content2 = st.session_state.results_df.loc[st.session_state.results_df['url'] == url2, 'content_preview'].values
                                     if len(content2)>0: st.text_area("Content preview 2:", content2[0], height=150, key=f"cann_p2_{i}")


    # Internal Semantic Search Tab (Dynamic Index)
    if "Internal Semantic Search" in tab_titles:
        semantic_search_tab_index = tab_titles.index("Internal Semantic Search")
        with tabs[semantic_search_tab_index]:
            st.subheader("Internal Semantic Search")
            
            # Only show if we have content data from main analysis
            if st.session_state.results_df is not None and st.session_state.content_dict is not None:
                
                # Helper function to log messages to UI
                def log_to_ui(message):
                    st.write(message)
                
                # Section 1: Index Management
                with st.expander("Step 1: Create or Load a Semantic Index", expanded=True):
                    st.write("Create a local vector database from your analyzed content for semantic search capabilities.")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Database path input with safe default
                        db_path_input = st.text_input(
                            "Database Path:",
                            value=st.session_state.search_db_path or os.path.join(os.getcwd(), "semantic_indexes"),
                            help="Directory where the vector database will be stored"
                        )
                        
                        collection_name_input = st.text_input(
                            "Collection Name:",
                            value=st.session_state.search_collection_name,
                            help="Name for this collection (useful for organizing multiple indexes)"
                        )
                    
                    with col2:
                        chunk_size = st.slider(
                            "Chunk Size:",
                            min_value=100,
                            max_value=2000,
                            value=500,
                            step=50,
                            help="Size of text chunks for indexing (smaller = more precise, larger = more context)"
                        )
                        
                        chunk_overlap = st.slider(
                            "Chunk Overlap:",
                            min_value=0,
                            max_value=200,
                            value=50,
                            step=10,
                            help="Overlap between chunks to maintain context"
                        )
                    
                    # Index status indicator
                    if st.session_state.search_index_exists:
                        if st.session_state.search_db_path and st.session_state.search_collection_name:
                            stats = get_index_stats(st.session_state.search_db_path, st.session_state.search_collection_name)
                            if stats:
                                st.success(f"✅ Index loaded: {stats.get('total_chunks', 0)} chunks from {stats.get('unique_urls', 0)} URLs")
                            else:
                                st.warning("⚠️ Index status unclear - try reloading")
                    else:
                        st.info("ℹ️ No semantic index currently loaded")
                    
                    # Index management buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Create/Update Index", use_container_width=True):
                            with st.spinner("Creating semantic index..."):
                                # Update session state
                                st.session_state.search_db_path = db_path_input
                                st.session_state.search_collection_name = collection_name_input
                                
                                # Get Jina API key
                                jina_api_key = os.getenv("JINA_API_KEY")
                                if not jina_api_key:
                                    st.error("JINA_API_KEY not found in environment variables")
                                else:
                                    # Prepare content data
                                    content_data = []
                                    for url, content in st.session_state.content_dict.items():
                                        if content and content.strip():
                                            content_data.append({
                                                'url': url,
                                                'content': str(content)
                                            })
                                    
                                    if content_data:
                                        # Create the index
                                        success = create_semantic_index(
                                            content_data=content_data,
                                            db_path=db_path_input,
                                            collection_name=collection_name_input,
                                            chunk_size=chunk_size,
                                            chunk_overlap=chunk_overlap,
                                            jina_api_key=jina_api_key,
                                            logger_callback=log_to_ui
                                        )
                                        
                                        if success:
                                            st.session_state.search_index_exists = True
                                            st.rerun()
                                        else:
                                            st.error("Failed to create semantic index")
                                    else:
                                        st.error("No content data available for indexing")
                    
                    with col2:
                        if st.button("Load Existing Index", use_container_width=True):
                            # Update session state and check if index exists
                            st.session_state.search_db_path = db_path_input
                            st.session_state.search_collection_name = collection_name_input
                            
                            exists = check_index_exists(db_path_input, collection_name_input)
                            st.session_state.search_index_exists = exists
                            
                            if exists:
                                st.success("Index loaded successfully!")
                                st.rerun()
                            else:
                                st.error("No index found at the specified location")

                # Section 2: Search Interface
                if st.session_state.search_index_exists:
                    st.header("Step 2: Query the Index")
                    
                    # Query mode selector
                    query_mode = st.radio(
                        "Select Query Mode:",
                        ["Single Query", "Batch Query (Multiple Queries)"],
                        horizontal=True,
                        key="query_mode"
                    )
                    
                    # Dynamic query input based on mode
                    if query_mode == "Single Query":
                        query_input = st.text_area(
                            "Enter your semantic query:",
                            height=100,
                            placeholder="e.g., 'How to optimize page load speed?'"
                        )
                    else:  # Batch Query
                        query_input = st.text_area(
                            "Enter your queries (one per line):",
                            height=250,
                            placeholder="e.g.:\nHow to improve SEO rankings?\nWhat are the best practices for content marketing?\nHow to optimize conversion rates?"
                        )
                    
                    # Search configuration
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        top_k = st.slider(
                            "Candidates to retrieve (top_k):",
                            min_value=5,
                            max_value=50,
                            value=20,
                            help="Number of documents to retrieve before re-ranking"
                        )
                    
                    with col2:
                        reranker_model = st.selectbox(
                            "Reranker Model:",
                            ["jina-reranker-m0", "jina-reranker-v2-base-multilingual"],
                            help="Model used for re-ranking search results"
                        )
                    
                    # Search button
                    if st.button("Search", key="semantic_search_button", use_container_width=True):
                        if not query_input or not query_input.strip():
                            st.error("Please enter a query")
                        else:
                            jina_api_key = os.getenv("JINA_API_KEY")
                            if not jina_api_key:
                                st.error("JINA_API_KEY not found in environment variables")
                            else:
                                with st.spinner("Searching..."):
                                    if query_mode == "Single Query":
                                        # Single query logic
                                        results = query_semantic_index(
                                            query_text=query_input.strip(),
                                            db_path=st.session_state.search_db_path,
                                            collection_name=st.session_state.search_collection_name,
                                            jina_api_key=jina_api_key,
                                            top_k=top_k,
                                            reranker_model_name=reranker_model,
                                            logger_callback=log_to_ui
                                        )
                                        st.session_state.search_results = results
                                        
                                    else:  # Batch Query
                                        # Batch query logic
                                        queries = [q.strip() for q in query_input.split('\n') if q.strip()]
                                        if not queries:
                                            st.error("No valid queries found")
                                        else:
                                            batch_results = {}
                                            status_text = st.empty()
                                            progress_bar = st.progress(0)
                                            total_queries = len(queries)
                                            
                                            for i, query in enumerate(queries):
                                                status_text.text(f"Processing query {i+1}/{total_queries}: '{query[:50]}...'")
                                                
                                                results_for_query = query_semantic_index(
                                                    query_text=query,
                                                    db_path=st.session_state.search_db_path,
                                                    collection_name=st.session_state.search_collection_name,
                                                    jina_api_key=jina_api_key,
                                                    top_k=top_k,
                                                    reranker_model_name=reranker_model,
                                                    logger_callback=lambda msg: None  # Silent for batch
                                                )
                                                
                                                batch_results[query] = results_for_query
                                                time.sleep(0.5)  # Rate limiting
                                                progress_bar.progress((i + 1) / total_queries)
                                            
                                            status_text.success(f"Batch processing complete for {total_queries} queries!")
                                            st.session_state.search_results = batch_results
                                            st.rerun()

                    # Display search results
                    if st.session_state.search_results:
                        st.header("Search Results")
                        
                        # Check if results are from single or batch query
                        if isinstance(st.session_state.search_results, list):
                            # Single query results
                            if st.session_state.search_results:
                                # Export button for single query
                                if st.button("Export Results to CSV", key="single_export"):
                                    df_data = []
                                    for result in st.session_state.search_results:
                                        df_data.append({
                                            'Rank': result['rank'],
                                            'Relevance Score': result['relevance_score'],
                                            'Source URL': result['source_url'],
                                            'Content Preview': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
                                            'Full Content': result['content']
                                        })
                                    
                                    df = pd.DataFrame(df_data)
                                    csv = df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="semantic_search_results.csv">Download CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                
                                # Display results
                                for i, result in enumerate(st.session_state.search_results):
                                    with st.expander(f"Rank {result['rank']}: {result['source_url']} (Score: {result['relevance_score']:.3f})"):
                                        st.write("**Content:**")
                                        st.write(result['content'])
                                        st.write(f"**Chunk:** {result.get('chunk_index', 0) + 1}")
                            else:
                                st.info("No results found for your query.")
                        
                        elif isinstance(st.session_state.search_results, dict):
                            # Batch query results
                            st.subheader("Batch Query Results")
                            
                            # Export button for batch query
                            if st.button("Export All Results to CSV", key="batch_export"):
                                df_data = []
                                for query, results in st.session_state.search_results.items():
                                    for result in results:
                                        df_data.append({
                                            'Query': query,
                                            'Rank': result['rank'],
                                            'Relevance Score': result['relevance_score'],
                                            'Source URL': result['source_url'],
                                            'Content Preview': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
                                            'Full Content': result['content']
                                        })
                                
                                if df_data:
                                    df = pd.DataFrame(df_data)
                                    csv = df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_semantic_search_results.csv">Download CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                            
                            # Display results for each query
                            for query, results in st.session_state.search_results.items():
                                with st.expander(f"Results for: '{query}' ({len(results)} found)"):
                                    if not results:
                                        st.info("No relevant documents found for this query.")
                                    else:
                                        for result in results:
                                            st.write(f"**Rank {result['rank']}** (Score: {result['relevance_score']:.3f})")
                                            st.write(f"**Source:** {result['source_url']}")
                                            st.write(f"**Content:** {result['content']}")
                                            st.divider()
                
                else:
                    st.info("🔍 Create or load a semantic index first to enable search functionality.")
            
            else:
                st.info("⚠️ Run the main analysis first to extract content data for semantic search.")

    # Content Inspector Tab (Dynamic Index)
    if "Content Inspector" in tab_titles:
        content_tab_index = tab_titles.index("Content Inspector")
        with tabs[content_tab_index]:
            st.subheader("Content Inspector")
            if not st.session_state.content_dict:
                st.info("No content was extracted or loaded. Run analysis with content extraction or load a file containing content.")
            else:
                st.markdown("Examine extracted content used for vectorization...")
                # Filters
                inspector_cols = st.columns([2, 1, 1])
                with inspector_cols[0]: url_search = st.text_input("Search URLs:", key="inspector_search")
                with inspector_cols[1]:
                     all_page_types_insp = ["All"] + sorted(st.session_state.results_df['page_type'].unique().tolist()) if 'page_type' in st.session_state.results_df.columns else ["All"]
                     inspector_page_type = st.selectbox("Filter by Page Type:", all_page_types_insp, key="inspector_type_filter")
                with inspector_cols[2]:
                     all_sitemaps_insp = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist()) if 'source_sitemap' in st.session_state.results_df.columns else ["All"]
                     inspector_sitemap = st.selectbox("Filter by Sitemap:", all_sitemaps_insp, key="inspector_sitemap_filter", disabled=len(all_sitemaps_insp) <= 1)

                # Get URLs with content
                all_inspectable_urls = sorted(list(st.session_state.content_dict.keys()))

                # Filter URLs (Original logic, use the combined list)
                filtered_inspect_urls = all_inspectable_urls[:]
                if url_search: filtered_inspect_urls = [url for url in filtered_inspect_urls if url_search.lower() in url.lower()]
                # Ensure results_df exists and has the columns before filtering
                if st.session_state.results_df is not None:
                    if inspector_page_type != "All" and 'page_type' in st.session_state.results_df.columns:
                         urls_matching_type = st.session_state.results_df[st.session_state.results_df['page_type'] == inspector_page_type]['url'].tolist()
                         filtered_inspect_urls = [url for url in filtered_inspect_urls if url in urls_matching_type]
                    if inspector_sitemap != "All" and 'source_sitemap' in st.session_state.results_df.columns:
                         urls_matching_sitemap = st.session_state.results_df[st.session_state.results_df['source_sitemap'] == inspector_sitemap]['url'].tolist()
                         filtered_inspect_urls = [url for url in filtered_inspect_urls if url in urls_matching_sitemap]


                if filtered_inspect_urls:
                    selected_url_inspect = st.selectbox("Select URL to inspect:", filtered_inspect_urls, key="inspector_url_select")
                    if selected_url_inspect:
                        # Get data
                        raw_content = st.session_state.content_dict.get(selected_url_inspect, "N/A")
                        source_sitemap = st.session_state.url_sources.get(selected_url_inspect, "Unknown")

                        # Get info from results_df if available
                        distance, page_type, percentile = None, None, None
                        if st.session_state.results_df is not None and selected_url_inspect in st.session_state.results_df['url'].values:
                            url_row = st.session_state.results_df[st.session_state.results_df['url'] == selected_url_inspect].iloc[0]
                            distance = url_row.get('distance_from_centroid')
                            page_type = url_row.get('page_type')
                            # Percentile calculation
                            if distance is not None:
                                sorted_df = st.session_state.results_df.sort_values('distance_from_centroid').reset_index()
                                rank_list = sorted_df.index[sorted_df['url'] == selected_url_inspect].tolist()
                                if rank_list: percentile = (rank_list[0] / (len(sorted_df)-1)) * 100 if len(sorted_df) > 1 else 0

                        # Display info
                        st.subheader("URL Information")
                        info_cols=st.columns(4)
                        with info_cols[0]: st.markdown(f"**URL:** [{selected_url_inspect}]({selected_url_inspect})")
                        with info_cols[1]: st.markdown(f"**Sitemap:** {source_sitemap}")
                        with info_cols[2]: st.markdown(f"**Type:** {page_type or 'N/A'}")
                        with info_cols[3]:
                            if distance is not None: st.markdown(f"**Distance:** {distance:.3f}")
                            if percentile is not None: st.markdown(f"**Focus Rank:** {percentile:.1f}%")
                            if percentile is not None:
                                 if percentile < 20: st.success("Highly Focused")
                                 elif percentile > 80: st.warning("Highly Divergent")

                        # Display content
                        st.subheader("Content Used for Vectorization")
                        st.markdown("**Raw content** (used for Jina embeddings):")
                        st.text_area("Raw Content", str(raw_content), height=300, key="inspect_raw")
                        st.info(f"Length: {len(str(raw_content))} characters")
                        
                        # Content statistics
                        if raw_content != "N/A" and len(str(raw_content)) > 0:
                            with st.expander("Content Statistics"):
                                words = str(raw_content).split()
                                word_count = len(words)
                                char_count = len(str(raw_content))
                                stats_cols = st.columns(3)
                                with stats_cols[0]: st.metric("Characters", char_count)
                                with stats_cols[1]: st.metric("Words", word_count)
                                with stats_cols[2]: st.metric("Avg Word Length", f"{char_count/word_count:.1f}" if word_count > 0 else "0")

                else:
                    st.warning("No URLs match your filter criteria or no content available to inspect.")

                # Bulk export (simplified)
                with st.expander("Bulk Export Options"):
                    st.markdown("Download extracted content and analysis data as CSV")
                    if st.button("Generate Content CSV", key="export_csv_button"):
                        if not all_inspectable_urls:
                             st.warning("No content URLs available to export.")
                        else:
                             export_data = []
                             for url in all_inspectable_urls:
                                  raw = st.session_state.content_dict.get(url, "")
                                  source = st.session_state.url_sources.get(url, "Unknown")
                                  distance, page_type = None, None
                                  if st.session_state.results_df is not None and url in st.session_state.results_df['url'].values:
                                      url_row = st.session_state.results_df[st.session_state.results_df['url'] == url].iloc[0]
                                      distance = url_row.get('distance_from_centroid')
                                      page_type = url_row.get('page_type')

                                  export_data.append({
                                     'URL': url, 'Source': source, 'Type': page_type, 'Distance': distance,
                                     'Content_Length': len(str(raw)), 'Raw_Content': str(raw)
                                  })
                             export_df = pd.DataFrame(export_data)
                             csv = export_df.to_csv(index=False).encode('utf-8')
                             b64 = base64.b64encode(csv).decode()
                             href = f'<a href="data:file/csv;base64,{b64}" download="site_content_analysis.csv">Download Content Analysis CSV</a>'
                             st.markdown(href, unsafe_allow_html=True)

    # Interactive Topic Cluster Analysis Tab (Phase 2)
    if "Topic Cluster Analysis" in tab_titles:
        cluster_tab_index = tab_titles.index("Topic Cluster Analysis")
        with tabs[cluster_tab_index]:
            st.subheader("Interactive Cluster Calibration")
            
            # Check if Phase 1 is complete
            if st.session_state.embeddings_matrix is None:
                st.info("🔄 Complete Phase 1 analysis first to enable interactive clustering.")
            else:
                # Educational expandder with instructions
                with st.expander("How to Use These Parameters to Find Topic Clusters"):
                    st.markdown("""
                    This section allows you to group your URLs into strategic "Topic Clusters" based on their content similarity. Think of it as an automatic categorization of your website. You can fine-tune the clustering algorithm using the two parameters below.

                    ---

                    #### **1. Epsilon (eps): The 'Neighborhood' Radius**

                    *   **What it is:** This slider controls how "close" two pages need to be to be considered part of the same topic. It's like setting a personal space bubble around each URL.
                    *   **How it works:**
                        *   **LOWER `eps` value** (e.g., 0.15): The bubble is smaller. The algorithm becomes more strict and will only group pages that are *very* similar. This will result in **more, smaller, and highly specific clusters.**
                        *   **HIGHER `eps` value** (e.g., 0.50): The bubble is larger. The algorithm is more lenient and will group pages that are loosely related. This will result in **fewer, larger, and broader topic clusters.**

                    ---

                    #### **2. Min Samples: The 'Group' Threshold**

                    *   **What it is:** This sets the minimum number of pages required to form a "real" topic cluster. It's the minimum size for a group to be considered significant.
                    *   **How it works:**
                        *   **HIGHER `Min Samples` value** (e.g., 10): The algorithm will only identify major topic pillars that consist of at least 10 pages. Smaller groups will be ignored (classified as "noise"). This is useful for finding the main themes of a large site.
                        *   **LOWER `Min Samples` value** (e.g., 3): The algorithm will also identify smaller, niche topic groups. This is useful for discovering emerging topics or smaller content categories.

                    ---

                    **💡 Quick Tip:** Start with the default settings. If you get one giant cluster, **decrease `eps`** slightly and run the analysis again. If you get too many tiny, insignificant clusters, **increase `Min Samples`**. Experiment until the resulting clusters make strategic sense for your website.
                    """)
                
                # Parameter controls
                col1, col2 = st.columns(2)
                with col1:
                    eps_value = st.slider(
                        "Epsilon (eps): Neighborhood Radius", 
                        min_value=0.05, 
                        max_value=1.0, 
                        value=0.3, 
                        step=0.01, 
                        key="dbscan_eps",
                        help="Lower values = more strict clustering (more, smaller clusters)"
                    )
                with col2:
                    min_samples_value = st.slider(
                        "Min Samples: Group Threshold", 
                        min_value=2, 
                        max_value=20, 
                        value=3, 
                        key="dbscan_min_samples",
                        help="Higher values = only major topic groups will be identified"
                    )
                
                # Generate clusters button
                if st.button("Generate & Name Clusters", use_container_width=True, key="generate_clusters_button"):
                    success = perform_interactive_cluster_analysis(eps=eps_value, min_samples=min_samples_value)
                    if success:
                        st.rerun()  # Refresh to show results immediately
                
                # Display results if available
                if st.session_state.cluster_summary and len(st.session_state.cluster_summary) > 0:
                    st.markdown("---")
                    st.subheader("Cluster Results")
                    
                    # Display cluster summary
                    for cluster_data in st.session_state.cluster_summary:
                        cluster_id = cluster_data['cluster_id']
                        cluster_name = cluster_data['cluster_name']
                        url_count = cluster_data['url_count']
                        example_urls = cluster_data['example_urls']
                        
                        if cluster_id == -1:
                            # Special handling for outliers/noise
                            with st.expander(f"🔍 {cluster_name} ({url_count} URLs)", expanded=False):
                                st.markdown("These are individual pages with unique topics that don't cluster with other content.")
                        else:
                            with st.expander(f"📁 **{cluster_name}** ({url_count} URLs)", expanded=False):
                                st.markdown(f"**Cluster ID:** {cluster_id}")
                                st.markdown(f"**Number of URLs:** {url_count}")
                                st.markdown("**Example URLs:**")
                                for url in example_urls[:5]:  # Show max 5 examples
                                    st.markdown(f"• {url}")
                                
                                if url_count > 5:
                                    st.markdown(f"*... and {url_count - 5} more URLs*")
                    
                    # Summary statistics
                    total_clustered = sum(c['url_count'] for c in st.session_state.cluster_summary if c['cluster_id'] != -1)
                    total_outliers = sum(c['url_count'] for c in st.session_state.cluster_summary if c['cluster_id'] == -1)
                    total_clusters = len([c for c in st.session_state.cluster_summary if c['cluster_id'] != -1])
                    
                    st.markdown("---")
                    st.subheader("Clustering Summary")
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Topic Clusters Found", total_clusters)
                    with summary_cols[1]:
                        st.metric("URLs in Clusters", total_clustered)
                    with summary_cols[2]:
                        st.metric("Outlier URLs", total_outliers)
                    
                    # Strategic insights
                    if total_clusters > 0:
                        st.markdown("---")
                        st.subheader("Strategic Insights")
                        
                        if total_clusters >= 5:
                            st.success("🎯 **Well-diversified content**: Your site covers multiple distinct topics, which can help capture diverse search queries.")
                        elif total_clusters >= 3:
                            st.info("📊 **Moderate topic variety**: Good balance between focus and coverage.")
                        else:
                            st.warning("🔍 **Highly focused content**: Very specialized topic coverage - consider expanding if targeting broader audiences.")
                        
                        if total_outliers > total_clustered * 0.3:
                            st.warning("⚠️ **High content diversity**: Many unique topics detected. Consider creating more content around your main themes to strengthen topical authority.")
                        elif total_outliers < total_clustered * 0.1:
                            st.success("✅ **Strong topical coherence**: Most content fits into clear topic groups, indicating focused content strategy.")
                        
                        # Export functionality
                        st.markdown("---")
                        with st.expander("📄 Export Cluster Data"):
                            if st.button("Generate Cluster Analysis CSV", key="export_clusters"):
                                cluster_export_data = []
                                
                                # Get detailed cluster assignments from results_df
                                if 'Topic Cluster' in st.session_state.results_df.columns:
                                    for _, row in st.session_state.results_df.iterrows():
                                        url = row['url']
                                        cluster_name = row['Topic Cluster']
                                        
                                        # Find cluster ID from cluster_names
                                        cluster_id = -1
                                        if cluster_name != "Outliers/Noise":
                                            for cid, cname in st.session_state.cluster_names.items():
                                                if cname == cluster_name:
                                                    cluster_id = cid
                                                    break
                                        
                                        cluster_export_data.append({
                                            'URL': url,
                                            'Cluster_ID': cluster_id,
                                            'Cluster_Name': cluster_name,
                                            'Page_Type': row.get('page_type', 'N/A'),
                                            'Distance_From_Centroid': row.get('distance_from_centroid', 'N/A')
                                        })
                                    
                                    cluster_df = pd.DataFrame(cluster_export_data)
                                    csv = cluster_df.to_csv(index=False).encode('utf-8')
                                    b64 = base64.b64encode(csv).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="topic_cluster_analysis.csv">Download Topic Cluster Analysis CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                    st.success("✅ Cluster analysis CSV generated successfully!")
                                else:
                                    st.error("No cluster data found in results. Please generate clusters first.")
                else:
                    st.info("🛠️ Use the controls above to generate topic clusters from your analyzed content.")


# Footer (Keep as is)
st.markdown("---")
st.markdown("**Topical Focus Analyzer** | Built with Python, Streamlit, and Content Analysis")
# --- END OF FILE ---