import requests
from bs4 import BeautifulSoup
import re
import time
import random
from urllib.parse import urlparse
import concurrent.futures

# Imports for advanced content extraction (headless browser)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    # Additional imports for explicit waits
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

def extract_main_content(url, timeout=30, user_agent=None, num_blocks_to_combine=2):
    """
    Extract the main content from a webpage using BeautifulSoup with intelligent aggregation.
    
    Args:
        url (str): URL to extract content from
        timeout (int): Request timeout in seconds
        user_agent (str, optional): Custom user agent string
        num_blocks_to_combine (int): Number of largest content blocks to combine
        
    Returns:
        str: Extracted main content text or empty string if extraction failed
    """
    # Define a list of user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    
    # Use provided user agent or randomly select one
    headers = {
        'User-Agent': user_agent if user_agent else random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # --- Stage 1: Try Trafilatura first (best for clean articles) ---
        try:
            import trafilatura
            # Use a `no_fallback=True` setting to make it fail faster if it's not confident
            extracted_text = trafilatura.extract(response.text, no_fallback=True)
            if extracted_text and len(extracted_text) > 200:  # Add a length check for confidence
                return clean_text(extracted_text)
        except ImportError:
            pass  # Trafilatura not installed
        except Exception:
            pass  # Trafilatura failed to find content confidently

        # --- Stage 2: If Trafilatura fails, use our new prioritized BeautifulSoup engine ---
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Decompose unwanted elements BEFORE passing to the extraction engine
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Call our new, centralized BeautifulSoup logic with aggregation parameter
        extracted_text = _extract_content_with_beautifulsoup(soup, num_blocks_to_combine)
        
        return clean_text(extracted_text)
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def _extract_content_with_beautifulsoup(soup: BeautifulSoup, num_to_combine: int = 2) -> str:
    """
    Extracts content using a robust "gather, de-duplicate, then evaluate" strategy
    to handle nested elements and fragmented page layouts correctly.
    
    Args:
        soup (BeautifulSoup): Parsed HTML soup object
        num_to_combine (int): Number of largest content blocks to combine
        
    Returns:
        str: Aggregated content text from the top N matching elements
    """
    # --- Step 1: Gather ALL potential candidates ---
    candidate_selectors = [
        # High-confidence selectors for page builders
        "div.elementor-widget-theme-post-content",  # Elementor page builder
        "div.et_pb_post_content",                   # Divi theme builder
        "div#post-content",                         # Common theme pattern
        "div.td-post-content",                      # Newspaper theme and similar
        "div.entry-content",                        # Very common WordPress class
        "div.post-content",                         # Generic post content class
        "div.article-content",                      # Article-specific content class
        "div.content-area",                         # Content area container
        "div[data-elementor-type='post']",          # Elementor post container
        "div.wp-block-post-content",                # WordPress block editor
        # Semantic HTML5 tags
        "article",                                  # Semantic article tag
        "main",                                     # HTML5 main content container
        # Generic content selectors (broader search)
        "div[id*='content']",                       # DIVs with 'content' in ID
        "div[class*='content']",                    # DIVs with 'content' in class
        "div[id*='main']",                          # DIVs with 'main' in ID
        "div[class*='main']",                       # DIVs with 'main' in class
        "div[id*='post']",                          # DIVs with 'post' in ID
        "div[class*='post']",                       # DIVs with 'post' in class
        "div[class*='article']",                    # DIVs with 'article' in class
    ]
    
    all_candidates = []
    for selector in candidate_selectors:
        all_candidates.extend(soup.select(selector))
    
    if not all_candidates:
        # Fallback to body if no candidates found
        if soup.body:
            text = soup.body.get_text(separator=' ', strip=True)
            if text and len(text) > 150:
                return text
        return ""

    # --- Step 2: CRITICAL - De-duplicate and filter out nested candidates ---
    # This is the core of the bug fix. We ensure we don't have containers inside other containers.
    unique_top_level_candidates = []
    
    for candidate in all_candidates:
        is_nested = False
        
        # Check if this candidate is a child of any element already in our unique list
        for existing_parent in unique_top_level_candidates:
            if candidate in existing_parent.find_all(recursive=True):
                is_nested = True
                break
        
        if not is_nested:
            # This candidate is not a child of any existing unique candidate.
            # Now, check if it's a PARENT of any existing unique candidate.
            # If so, we must replace the child with this more complete parent.
            
            # Remove any elements that are children of the current candidate
            unique_top_level_candidates = [
                elem for elem in unique_top_level_candidates 
                if elem not in candidate.find_all(recursive=True)
            ]
            unique_top_level_candidates.append(candidate)

    # --- Step 3: Evaluate the clean, unique candidates ---
    evaluated_candidates = []
    MIN_TEXT_LENGTH = 100  # Lowered threshold since we now have proper de-duplication
    
    for candidate in unique_top_level_candidates:
        text = candidate.get_text(separator=' ', strip=True)
        current_length = len(text)
        
        # Only consider candidates that are sufficiently long
        if current_length > MIN_TEXT_LENGTH:
            evaluated_candidates.append({'text': text, 'length': current_length})
    
    if not evaluated_candidates:
        return ""

    # --- Step 4: Sort and aggregate the top N ---
    evaluated_candidates.sort(key=lambda x: x['length'], reverse=True)
    
    num_to_take = min(num_to_combine, len(evaluated_candidates))
    top_candidates = evaluated_candidates[:num_to_take]
    
    final_content = "\n\n".join([candidate['text'] for candidate in top_candidates])
    
    return final_content


def clean_text(text):
    """
    Clean the extracted text by removing extra whitespace, URLs, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def _extract_content_with_headless_browser(url: str, timeout: int = 45, num_blocks_to_combine: int = 2) -> str:
    """
    Extracts content from a URL by rendering it in a headless Chrome browser with intelligent aggregation.
    This is slower but handles JavaScript-rendered content.
    
    Args:
        url (str): URL to extract content from
        timeout (int): Browser timeout in seconds
        num_blocks_to_combine (int): Number of largest content blocks to combine
        
    Returns:
        str: Extracted main content text or empty string if extraction failed
    """
    if not SELENIUM_AVAILABLE:
        print("Selenium not available. Please install selenium and webdriver-manager.")
        return ""
    
    # Configure Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = None
    try:
        # Initialize the Chrome driver automatically
        # webdriver-manager will handle downloading/caching the correct driver
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set timeouts
        driver.set_page_load_timeout(timeout)
        driver.implicitly_wait(10)
        
        # Navigate to the URL
        driver.get(url)
        
        # Define prioritized list of CSS selectors that are common across well-structured pages
        PRIORITY_SELECTORS = [
            "article",                   # Highest priority: semantic <article> tag
            "div[itemprop='articleBody']", # High priority: Schema.org markup
            "main",                      # Common HTML5 main content container
            "h1"                         # Reliable fallback: the main page title
        ]
        
        # Instantiate WebDriverWait with generous timeout (30 seconds max wait time)
        wait = WebDriverWait(driver, 30)
        
        # Wait for the first available selector from our priority list
        combined_selector = ", ".join(PRIORITY_SELECTORS)
        
        try:
            # Wait for the first element from our priority list to appear in the DOM
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, combined_selector)))
            # If the code reaches here, it means content has likely rendered
            # Add a tiny extra sleep just in case, for any trailing scripts
            time.sleep(1)  # A small, final buffer
        except TimeoutException:
            # This block executes if NONE of our priority selectors were found within the 30-second timeout
            print(f"Content did not load for {url} within the time limit. Skipping.")
            if driver:
                driver.quit()
            return ""
        
        # Get the final, rendered page source (HTML)
        page_source = driver.page_source
        
    except Exception as e:
        print(f"Error during headless browser extraction for {url}: {e}")
        return ""
    finally:
        if driver:
            driver.quit()  # Crucial: always close the browser to free up resources
    
    # Now, process the rendered HTML using the same multi-stage extraction logic
    # This ensures consistent processing between standard and advanced extraction modes
    
    # --- Stage 1: Try Trafilatura first ---
    try:
        import trafilatura
        extracted_text = trafilatura.extract(page_source, no_fallback=True)
        if extracted_text and len(extracted_text) > 200:
            return clean_text(extracted_text)
    except Exception:
        pass

    # --- Stage 2: Fallback to our new BeautifulSoup engine ---
    soup = BeautifulSoup(page_source, 'lxml')
    
    # Decompose unwanted elements BEFORE passing to the extraction engine
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Call our new, centralized BeautifulSoup logic with aggregation parameter
    extracted_text = _extract_content_with_beautifulsoup(soup, num_blocks_to_combine)

    return clean_text(extracted_text)


def batch_extract_content(urls, max_workers=5, delay=1.0, use_advanced_extraction=False, num_blocks_to_combine=2):
    """
    Extract content from multiple URLs in parallel with rate limiting.
    Can now use standard or advanced (headless) extraction with intelligent content aggregation.
    
    Args:
        urls (list): List of URLs to extract content from
        max_workers (int): Maximum number of parallel workers
        delay (float): Delay between requests in seconds to avoid rate limiting
        use_advanced_extraction (bool): Use headless browser for JavaScript-heavy sites
        num_blocks_to_combine (int): Number of largest content blocks to combine per page
        
    Returns:
        dict: Dictionary mapping URLs to extracted content
    """
    results = {}
    domains_last_request = {}  # Track last request time per domain
    
    def extract_with_rate_limit(url):
        # Extract domain to apply rate limiting per domain
        domain = urlparse(url).netloc
        
        # Check if we need to wait before making another request to this domain
        if domain in domains_last_request:
            elapsed = time.time() - domains_last_request[domain]
            if elapsed < delay:
                time.sleep(delay - elapsed)
        
        # CORE LOGIC CHANGE: Choose the extraction function based on the parameters
        if use_advanced_extraction:
            content = _extract_content_with_headless_browser(url, num_blocks_to_combine=num_blocks_to_combine)
        else:
            # The original `extract_main_content` is now the default path
            content = extract_main_content(url, num_blocks_to_combine=num_blocks_to_combine)
        
        # Update last request time for this domain
        domains_last_request[domain] = time.time()
        
        return url, content
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_with_rate_limit, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                url, content = future.result()
                results[url] = content
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results[url] = ""
    
    return results

# Test function
def test_content_extractor():
    test_urls = [
        "https://www.python.org/about/",
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://www.bbc.com/news"
    ]
    
    for url in test_urls:
        print(f"\nTesting content extraction for: {url}")
        content = extract_main_content(url)
        
        # Print a preview
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Extracted content ({len(content)} characters):")
        print(preview)
        
        # Show content statistics  
        words = content.split()
        print(f"\nContent statistics: {len(words)} words, {len(content)} characters")

if __name__ == "__main__":
    test_content_extractor()