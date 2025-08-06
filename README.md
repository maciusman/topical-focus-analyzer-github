# 🎯 Topical Focus Analyzer

A comprehensive Streamlit application for **website content analysis** that performs content extraction, semantic vectorization, clustering, and visualization to measure **topical focus** and **content coherence** across multiple sitemaps.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Key Features

### 🔍 **Enhanced Multi-Stage Content Extraction**
- **Intelligent Page Builder Support**: Optimized for Elementor, Divi, WordPress blocks
- **Multi-block Content Aggregation**: Configurable 1-5 content blocks combining
- **Advanced Extraction Mode**: Optional Selenium-based rendering for JS-heavy sites
- **Robust De-duplication**: Eliminates parent-child content duplicates

### 📊 **Comprehensive Analysis Suite**
- **Focus Score Calculation**: Measures website topical coherence
- **Semantic Clustering**: DBSCAN-based topic grouping with LLM-generated names  
- **Content Cannibalization Detection**: Identifies overlapping/duplicate content
- **t-SNE Visualization**: 2D interactive content relationship mapping

### 🤖 **AI-Powered Features**
- **Internal Semantic Search**: Vector-based content querying with re-ranking
- **LLM Content Summarization**: OpenRouter integration for intelligent summaries
- **Smart Cluster Naming**: AI-generated topic cluster descriptions
- **Advanced Re-ranking**: Jina Reranker for improved search relevance

### 🎨 **Multi-Tab Interface**
1. **Overview** - Focus metrics, distributions, AI summaries
2. **URL Details** - Searchable/filterable results table with CSV export
3. **Visual Map** - Interactive t-SNE plots with multiple color schemes  
4. **Cannibalization Analysis** - Duplicate content detection and export
5. **Topic Cluster Analysis** - Interactive clustering with parameter tuning
6. **Internal Semantic Search** - Vector database creation and querying
7. **Content Inspector** - Raw content viewing and bulk export

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required API keys:
  - **Jina AI API key** (for embeddings and re-ranking)
  - **OpenRouter API key** (optional, for AI summaries)

### Installation

#### Option 1: Using Poetry (Recommended)
```bash
git clone https://github.com/yourusername/topical-focus-analyzer.git
cd topical-focus-analyzer
poetry install
poetry run streamlit run multi_sitemap_app.py
```

#### Option 2: Using pip
```bash
git clone https://github.com/yourusername/topical-focus-analyzer.git
cd topical-focus-analyzer
pip install -r requirements.txt
streamlit run multi_sitemap_app.py
```

#### Option 3: Windows Batch File
```bash
# Simply run the provided batch file
Start_Topical_Focus_Analyzer.bat
```

### Configuration

1. **Launch the application** - Navigate to `http://localhost:8501`
2. **Enter API keys** in the sidebar configuration panel
3. **Input your domain** (e.g., `example.com`)
4. **Configure analysis parameters**:
   - Content extraction mode (Standard/Advanced)
   - Number of content blocks to combine (1-5)
   - URL filtering patterns
   - Analysis depth settings

## 📖 Usage Guide

### Basic Workflow
1. **Enter target domain** in the main input field
2. **Configure extraction settings**:
   - Use **1-2 blocks** for precision (simple articles/blogs)
   - Use **3-4 blocks** for completeness (complex landing pages)
   - Enable **Advanced Mode** for JavaScript-heavy sites
3. **Apply URL filters** (include/exclude patterns)
4. **Start analysis** and monitor progress
5. **Explore results** across multiple tabs
6. **Export data** as needed (CSV, analysis states)

### Content Extraction Optimization

#### **Standard Mode (Default)**
- Fast BeautifulSoup-based extraction
- Perfect for most WordPress sites, blogs, content sites
- Supports modern page builders (Elementor, Divi)

#### **Advanced Mode**  
- Selenium-based rendering for JavaScript content
- Use for Single Page Applications (SPAs)
- Required for dynamic content loading

#### **Content Block Aggregation**
- **1 block**: Maximum precision, single best content section
- **2-3 blocks**: Balanced approach (recommended default)
- **4-5 blocks**: Maximum completeness for complex layouts

## 🏗️ Architecture

### Core Modules
- **`multi_sitemap_app.py`** - Main Streamlit application
- **`modules/content_extractor.py`** - Enhanced multi-stage content extraction
- **`modules/semantic_search_engine.py`** - Vector database and search functionality
- **`modules/analyzer.py`** - Focus score calculations and core metrics
- **`modules/clustering.py`** - DBSCAN clustering with LLM integration
- **`modules/simple_vectorizer.py`** - Jina embeddings generation
- **`modules/llm_summarizer.py`** - OpenRouter LLM integration

### Key Technologies
- **Streamlit** - Web application framework
- **Jina AI** - Embeddings and re-ranking
- **ChromaDB** - Vector database for semantic search
- **OpenRouter** - LLM API gateway
- **Plotly** - Interactive visualizations
- **BeautifulSoup/Selenium** - Content extraction
- **scikit-learn** - Clustering and dimensionality reduction

## 🔧 Advanced Features

### Semantic Search Engine
- **Local ChromaDB** vector database creation
- **Configurable chunking** (size, overlap parameters)
- **Batch query processing** for bulk searches
- **Advanced re-ranking** with Jina models
- **Persistent indexes** with statistics tracking

### Content Inspector
- **Raw content viewing** with search and filtering
- **Full content export** (not just previews)
- **Content statistics** and quality metrics
- **Bulk operations** for large datasets

### Cannibalization Detection
- **Similarity threshold** configuration  
- **Detailed content comparison** side-by-side
- **Export functionality** for duplicate content analysis
- **Strategic recommendations** for content optimization

## 📊 Output Formats

### Export Options
- **CSV exports** for all major data types
- **Pickled analysis states** for session resumption  
- **Full content dumps** with metadata
- **Cluster analysis reports** with AI insights

### Visualization Outputs
- **Interactive t-SNE plots** with multiple color schemes
- **Content distribution charts** by sitemap/type
- **Distance distribution histograms**
- **Focus score visualizations**

## 🛡️ Security & Privacy

- **No data storage** - all processing is local
- **API key protection** - keys stored in session state only
- **Safe content extraction** - respects robots.txt and rate limits
- **Privacy-focused** - no external data sharing beyond API calls

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, bug reports, or feature requests.

### Development Setup
```bash
git clone https://github.com/yourusername/topical-focus-analyzer.git
cd topical-focus-analyzer
poetry install --with dev
poetry run streamlit run multi_sitemap_app.py
```

## 📚 Documentation

For detailed code documentation and implementation details, see:
- **[MULTI_SITEMAP_APP_CODE_MAP.md](MULTI_SITEMAP_APP_CODE_MAP.md)** - Complete code structure and feature documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Jina AI** for embeddings and re-ranking capabilities
- **OpenRouter** for LLM API access  
- **Streamlit** for the amazing web app framework
- **ChromaDB** for vector database functionality

---

**Latest Enhancement (August 2025)**: Enhanced Multi-Stage Content Extraction Engine with Intelligent Content Aggregation and robust de-duplication logic for modern page builders.