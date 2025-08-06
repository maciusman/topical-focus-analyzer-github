import os
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Optional, Callable

# Load environment variables
load_dotenv()

@st.cache_data
def fetch_openrouter_models(api_key: str) -> Optional[List[Dict]]:
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key: OpenRouter API key
        
    Returns:
        List of model dictionaries or None if failed
    """
    if not api_key:
        return None
        
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result:
                # Filter for models that are likely good for summarization tasks
                # and sort by popularity/capability
                models = result["data"]
                
                # Group models by provider but don't heavily prioritize any single one
                preferred_prefixes = [
                    "anthropic/",      # Claude models first
                    "openai/",         # OpenAI models
                    "mistralai/",      # Mistral models  
                    "meta-llama/",     # Llama models
                    "google/",         # Google models
                    "cohere/",         # Cohere models
                    "perplexity/",     # Perplexity models
                ]
                
                # Sort models: mix preferred providers, then others alphabetically
                def sort_key(model):
                    name = model.get("id", "")
                    # First, check if it's a preferred provider
                    for i, prefix in enumerate(preferred_prefixes):
                        if name.startswith(prefix):
                            return (0, i, name)  # Group 0 for preferred, then by provider order
                    # All other models go to group 1
                    return (1, name)
                
                models.sort(key=sort_key)
                return models
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
        return None

def generate_summary_with_openrouter(api_key: str, model_id: str, focus_score: float, 
                                   radius_score: float, total_urls: int, top_focused_urls: List[str], 
                                   top_divergent_urls: List[str], page_type_distribution: Optional[Dict] = None,
                                   logger_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Generate a summary using OpenRouter API.
    
    Args:
        api_key: OpenRouter API key
        model_id: Model identifier (e.g., "mistralai/mistral-7b-instruct")
        focus_score: Site focus score (0-100)
        radius_score: Site radius score (0-100)
        total_urls: Total number of URLs analyzed
        top_focused_urls: List of most focused URLs
        top_divergent_urls: List of most divergent URLs
        page_type_distribution: Optional dictionary of page type counts
        logger_callback: Optional function to log messages to UI
        
    Returns:
        Generated summary text or None if failed
    """
    if not api_key:
        if logger_callback:
            logger_callback("❌ OpenRouter API key not provided")
        return None
        
    if logger_callback:
        logger_callback("🤖 Generating AI summary with OpenRouter...")
    
    # Format URL lists for the prompt
    def format_url_list(urls, max_urls=5):
        formatted_list = ""
        for i, url in enumerate(urls[:max_urls], 1):
            formatted_list += f"{i}. {url}\n"
        
        if len(urls) > max_urls:
            formatted_list += f"... and {len(urls) - max_urls} more\n"
        
        return formatted_list
    
    # Build the analysis prompt
    prompt_content = f"""
**WEBSITE CONTENT ANALYSIS DATA:**

• **Total URLs Analyzed:** {total_urls}
• **Site Focus Score:** {focus_score:.2f} / 100
  (How tightly connected the content is to a central theme. Higher = more focused)
• **Site Radius Score:** {radius_score:.2f} / 100
  (Breadth of topics covered relative to main theme. Higher = wider variety)

• **Most Focused URLs (closest to core theme):**
{format_url_list(top_focused_urls)}

• **Most Divergent URLs (furthest from core theme):**
{format_url_list(top_divergent_urls)}
"""

    if page_type_distribution:
        prompt_content += "\n• **Page Type Distribution:**\n"
        for page_type, count in page_type_distribution.items():
            percentage = (count / total_urls) * 100 if total_urls > 0 else 0
            prompt_content += f"  - {page_type}: {count} pages ({percentage:.1f}%)\n"

    prompt_content += """

**ANALYSIS REQUIRED:**

Provide a clear, actionable analysis for the website owner including:

1. **Executive Summary**: What do the focus and radius scores mean for this specific website?

2. **Key Insights**: What does the URL data reveal about the site's content strategy and topical coherence?

3. **Action Plan**: 2-4 specific, actionable recommendations to improve or maintain the site's topical focus.

Keep the language clear and practical, avoiding overly technical jargon. Base your analysis strictly on the provided data.
"""

    try:
        # Prepare the request payload in OpenAI format
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert Website Content Strategist and SEO Analyst. Provide clear, actionable insights based on topical analysis data."
                },
                {
                    "role": "user", 
                    "content": prompt_content
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/topical-focus-analyzer",  # Optional but recommended
            "X-Title": "Topical Focus Analyzer"  # Optional app identification
        }
        
        if logger_callback:
            logger_callback(f"📡 Sending request to {model_id}...")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                summary = message.get("content", "").strip()
                
                if summary:
                    if logger_callback:
                        logger_callback("✅ AI summary generated successfully")
                    return summary
                else:
                    if logger_callback:
                        logger_callback("❌ Empty response from AI model")
                    return None
            else:
                if logger_callback:
                    logger_callback("❌ No choices in AI response")
                return None
                
        elif response.status_code == 429:
            if logger_callback:
                logger_callback("⚠️ Rate limit hit - please try again later")
            return None
            
        elif response.status_code == 401:
            if logger_callback:
                logger_callback("❌ Invalid API key")
            return None
            
        else:
            if logger_callback:
                logger_callback(f"❌ API error {response.status_code}: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        if logger_callback:
            logger_callback("❌ Request timeout - model may be overloaded")
        return None
    except requests.exceptions.RequestException as e:
        if logger_callback:
            logger_callback(f"❌ Request error: {str(e)}")
        return None
    except Exception as e:
        if logger_callback:
            logger_callback(f"❌ Unexpected error: {str(e)}")
        return None

# Legacy function name for compatibility
def get_gemini_summary(api_key, focus_score, radius_score, total_urls, top_focused_urls, 
                      top_divergent_urls, page_type_distribution=None, model_id=None, logger_callback=None):
    """
    Legacy wrapper function for compatibility with existing code.
    Now uses OpenRouter instead of Gemini.
    """
    # Default model if none specified
    if not model_id:
        model_id = "mistralai/mistral-7b-instruct"
    
    return generate_summary_with_openrouter(
        api_key=api_key,
        model_id=model_id,
        focus_score=focus_score,
        radius_score=radius_score,
        total_urls=total_urls,
        top_focused_urls=top_focused_urls,
        top_divergent_urls=top_divergent_urls,
        page_type_distribution=page_type_distribution,
        logger_callback=logger_callback
    )

# Test function
def test_openrouter_summarizer():
    """Test function for OpenRouter summarizer (requires valid API key)"""
    
    def dummy_logger(message):
        print(f"LOG: {message}")
    
    # Test data
    test_data = {
        'focus_score': 78.5,
        'radius_score': 45.2,
        'total_urls': 125,
        'top_focused_urls': [
            'https://example.com/products/widgets/blue-widget',
            'https://example.com/products/widgets/red-widget',
            'https://example.com/products/widgets/heavy-duty-widget',
            'https://example.com/services/widget-installation',
            'https://example.com/products/widgets/'
        ],
        'top_divergent_urls': [
            'https://example.com/blog/company-picnic-2023',
            'https://example.com/about-us/team/ceo',
            'https://example.com/contact',
            'https://example.com/careers/open-positions',
            'https://example.com/blog/unrelated-industry-news'
        ],
        'page_type_distribution': {
            'Product': 80,
            'Service': 5,
            'Blog': 15,
            'Informational': 10,
            'Category': 15
        }
    }
    
    # Try to get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        print("Testing OpenRouter summarizer...")
        
        # First, fetch available models
        models = fetch_openrouter_models(api_key)
        if models:
            print(f"Found {len(models)} available models")
            # Use the first available model for testing
            model_id = models[0]["id"]
            print(f"Testing with model: {model_id}")
        else:
            model_id = "mistralai/mistral-7b-instruct"  # Fallback
            print(f"Using fallback model: {model_id}")
        
        summary = generate_summary_with_openrouter(
            api_key=api_key,
            model_id=model_id,
            focus_score=test_data['focus_score'],
            radius_score=test_data['radius_score'],
            total_urls=test_data['total_urls'],
            top_focused_urls=test_data['top_focused_urls'],
            top_divergent_urls=test_data['top_divergent_urls'],
            page_type_distribution=test_data['page_type_distribution'],
            logger_callback=dummy_logger
        )
        
        if summary:
            print("\n--- Generated Summary ---")
            print(summary)
            print("--- End of Summary ---")
        else:
            print("Failed to generate summary")
    else:
        print("\n--- WARNING ---")
        print("No OPENROUTER_API_KEY found in environment variables (.env file).")
        print("Please set the OPENROUTER_API_KEY to test the summarizer function.")
        print("---------------")

def generate_cluster_names_with_openrouter(api_key: str, model_id: str, cluster_prompt: str,
                                         logger_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Generate cluster names using OpenRouter API with a custom prompt.
    
    Args:
        api_key: OpenRouter API key
        model_id: Model identifier (e.g., "anthropic/claude-3-sonnet")
        cluster_prompt: Custom prompt for cluster naming
        logger_callback: Optional callback for logging
        
    Returns:
        Raw response string from the LLM or None if failed
    """
    if not api_key or not model_id or not cluster_prompt:
        if logger_callback:
            logger_callback("❌ Missing required parameters for cluster naming")
        return None
        
    if logger_callback:
        logger_callback("🤖 Requesting cluster names from AI...")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://topical-focus-analyzer.streamlit.app",
                "X-Title": "Topical Focus Analyzer",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": cluster_prompt}],
                "temperature": 0.3,  # Lower temperature for more consistent naming
                "max_tokens": 1000   # Reasonable limit for cluster names
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                if logger_callback:
                    logger_callback("✅ Received cluster names from AI")
                return content
            else:
                if logger_callback:
                    logger_callback("❌ Invalid response format from OpenRouter for cluster naming")
                return None
        else:
            if logger_callback:
                logger_callback(f"❌ OpenRouter API error for cluster naming: {response.status_code}")
            return None
            
    except Exception as e:
        if logger_callback:
            logger_callback(f"❌ Error calling OpenRouter for cluster naming: {str(e)}")
        return None


if __name__ == "__main__":
    test_openrouter_summarizer()