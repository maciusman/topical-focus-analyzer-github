# Multi Sitemap App - Code Map & Documentation

  Dla szybkiej nawigacji:

- 🔍 Potrzebujesz CSV export? → Linia 607
- 🎨 Chcesz edytować UI layout? → Linie 329-595
- ⚙️ Modyfikujesz clustering? → Linie 37-156
- 📊 Zmieniasz visualizations? → Linie 1358-1414
- 🆕 **Content Extraction Options slider** → Linia ~460
- 🆕 **Enhanced extraction logic** → modules/content_extractor.py

## File Summary
A comprehensive Streamlit application for topical content analysis that performs website content extraction, vectorization, semantic clustering, and visualization with multi-tab interface for detailed analysis results.

**🆕 Latest Enhancement (Aug 2025):** Enhanced Multi-Stage Content Extraction Engine with Intelligent Content Aggregation and robust de-duplication logic for modern page builders (Elementor, Divi) and complex multi-section layouts.

## Key Dependencies (Top 5)
1. **streamlit** - Main web application framework for UI components and interactivity
2. **pandas** - Data manipulation and analysis for URL/content processing
3. **plotly** - Interactive visualizations (t-SNE plots, charts, graphs)
4. **numpy** - Numerical operations for embeddings and distance calculations
5. **pickle** - State persistence for saving/loading analysis sessions

## Complete Structure Index

### Core Functions
| Function | Line | Purpose |
|----------|------|---------|
| `perform_interactive_cluster_analysis(eps, min_samples)` | 37 | Interactive DBSCAN clustering with LLM-generated cluster names |
| `load_analysis_state(uploaded_file)` | 214 | Load saved analysis state from pickle file |
| `save_analysis_state(filename)` | 265 | Save current analysis state to pickle file |
| `log_to_ui(message)` | 599 | Add messages to UI logging system |
| `export_cannibalization_to_csv(displayed_duplicates)` | 607 | Export duplicate content analysis to CSV |

### Session State Management
| Section | Line Range | Purpose |
|---------|------------|---------|
| `DEFAULT_SESSION_STATE` definition | 163-207 | Complete default state structure with 25+ variables |
| Session state initialization | 209-211 | Initialize all session state variables |
| State loading/saving utilities | 214-328 | Pickle-based state persistence system |

## Application Structure by Logical Sections

### 1. Initialization & Setup (Lines 1-35)
- **Imports**: Standard libraries + custom modules from `/modules/` directory
- **Environment**: Load .env variables for API keys
- **Error handling**: Module import validation with user-friendly errors

### 2. Core Analysis Functions (Lines 36-328)
- **Interactive Clustering** (37-156): DBSCAN clustering with LLM naming
- **State Management** (158-328): Session state defaults, load/save functionality

### 3. Main UI Structure (Lines 329-595)
- **App Title** (331): Main header with site branding
- **Load Analysis Section** (340-352): Upload previous analysis files
- **Configuration Panel** (355-560):
  - API Configuration (358-386): OpenRouter, Jina API keys
  - URL Filtering (388-422): Include/exclude patterns with logic options
  - Content Analysis Options (424-560): Scraping parameters, AI settings
  - 🆕 **Content Extraction Options** (~460): Advanced extraction mode + **Intelligent Aggregation slider**
- **Save Analysis Section** (563-576): Export current state
- **Analysis Execution** (581-595): URL input and start button

### 4. Analysis Execution Engine (Lines 683-1138)
- **Input Processing** (683-790): URL validation, filtering application
- **Content Extraction** (791-950): 🆕 **Enhanced batch scraping** with intelligent aggregation + progress tracking
- **Vectorization** (951-1050): Jina embeddings generation
- **Analysis Calculations** (1051-1138): Focus scores, t-SNE, clustering

### 5. Results Display System (Lines 1139-2096)

#### Tab Structure (Lines 1148-1161)
```
Base tabs: ["Overview", "URL Details", "Visual Map (t-SNE)", "Cannibalization/Clusters"]
Dynamic tabs added based on data availability:
- "Topic Cluster Analysis" (if embeddings available)
- "Internal Semantic Search" (if content available)  
- "Content Inspector" (if content available)
```

#### Tab Implementations
| Tab | Line Range | Key Features |
|-----|------------|--------------|
| **Overview** (Tab 0) | 1164-1211 | Focus/radius scores, sitemap distribution, LLM summary, page type analysis |
| **URL Details** (Tab 1) | 1212-1357 | Complete results table with sorting, filtering, and export functionality |
| **Visual Map** (Tab 2) | 1358-1414 | Interactive t-SNE scatter plot with color coding options |
| **Cannibalization** (Tab 3) | 1415-1516 | Duplicate detection with similarity thresholds and CSV export |
| **Semantic Search** (Tab 4) | 1518-1817 | Vector database querying with relevance scoring |
| **Content Inspector** (Tab 5) | 1818-1931 | Raw content viewing and analysis |
| **Topic Clusters** (Tab 6) | 1932-2091 | Interactive DBSCAN clustering with parameter tuning |

### 6. Export & Utility Functions (Lines 599-681)
- **UI Logging System** (599-605): Message tracking for user feedback
- **CSV Export Functions** (607-681): Cannibalization and cluster analysis exports

## Key Features by Category

### 🔍 Content Analysis
- **Batch URL Processing**: Concurrent scraping with configurable workers
- **🆕 Enhanced Multi-Stage Content Extraction**: Intelligent page builder support (Elementor, Divi)
- **🆕 Intelligent Content Aggregation**: User-configurable multi-block content combining (1-5 blocks)
- **🆕 Robust De-duplication**: Parent-child relationship detection eliminates content duplicates
- **Advanced Content Extraction**: Optional Selenium-based JS rendering for complex sites
- **Jina Embeddings**: Vector generation for semantic analysis
- **Focus Score Calculation**: Website topical coherence measurement

### 📊 Visualization Components
- **t-SNE Scatter Plots**: 2D visualization of content relationships
- **Pie Charts**: Content distribution by source, page type
- **Interactive Tables**: Sortable, filterable results display
- **Cluster Visualizations**: Topic group analysis with color coding

### 🤖 AI Integration
- **OpenRouter LLM**: Cluster naming and content summarization
- **Multiple Model Support**: Dynamic model selection from API
- **Semantic Search**: Vector similarity matching
- **Smart Clustering**: DBSCAN with intelligent parameter suggestions

### 💾 State Management
- **Persistent Sessions**: Full analysis state preservation
- **Incremental Loading**: Resume analysis from saved checkpoints
- **Export Functionality**: CSV downloads for all major data types
- **Error Recovery**: Graceful handling of incomplete states

### 🎛️ User Interface
- **Multi-Tab Layout**: Organized feature separation
- **Dynamic Tab Loading**: Conditional tab display based on data availability
- **Progress Tracking**: Real-time feedback during long operations
- **Parameter Tuning**: Interactive controls for analysis customization
- **🆕 Intelligent Aggregation Control**: User-friendly slider for content block combination (1-5 blocks)
- **🆕 Advanced Extraction Toggle**: Seamless integration of standard/advanced extraction modes

## Session State Variables (26 Total)

### Analysis Results
- `sitemaps`, `selected_sitemaps`, `urls`, `url_sources`
- `content_dict`, `results_df`, `focus_score`, `radius_score`
- `pairwise_distances`, `llm_summary`, `centroid`

### Processing State
- `content_for_embedding`, `analysis_loaded`, `_loading_in_progress`
- `log_messages`, `embeddings_matrix`

### Clustering (Phase 2)
- `cluster_labels`, `cluster_names`, `cluster_summary`

### Semantic Search
- `search_db_path`, `search_collection_name`, `search_index_exists`, `search_results`

### User Input Parameters (13 Total)
- `domain`, `input_include_filters`, `input_exclude_filters`
- `input_include_logic_any`, `input_analyze_content`, `input_max_workers`
- `input_request_delay`, `input_advanced_extraction`, `input_max_urls`
- `input_perplexity`, `input_focus_k`, `input_radius_k`
- `input_use_ai_summary`, `input_jina_api_key`, `input_openrouter_api_key`, `input_selected_model`
- **🆕** `input_num_blocks_to_combine` - Intelligent content aggregation level (1-5)

## Architecture Notes

### Modular Design
- Separates concerns into `/modules/` directory
- Clean import structure with error handling
- Standardized function interfaces across modules
- **🆕 Enhanced content_extractor.py**: Multi-stage extraction with de-duplication logic

### Performance Optimizations
- Concurrent content extraction with worker pools
- Embeddings caching for interactive clustering
- Incremental state loading to avoid recomputation
- **🆕 Intelligent DOM parsing**: Efficient parent-child relationship detection
- **🆕 Smart content filtering**: Optimized MIN_TEXT_LENGTH thresholds (100 chars)

### User Experience
- Comprehensive progress feedback
- Detailed error messages with recovery suggestions
- Intuitive parameter defaults with help tooltips
- Persistent state across sessions

---

## 🆕 **Latest Updates (August 1, 2025)**

### **Enhanced Multi-Stage Content Extraction Engine**
- **Location**: `modules/content_extractor.py`
- **Key Function**: `_extract_content_with_beautifulsoup(soup, num_to_combine=2)`
- **Algorithm**: Gather → De-duplicate → Filter → Sort → Aggregate
- **Supported Builders**: Elementor, Divi, WordPress blocks, semantic HTML5

### **Intelligent Content Aggregation UI**
- **Location**: Line ~460 in main UI structure
- **Component**: Streamlit slider (1-5 blocks)
- **Integration**: Full parameter passing chain to extraction backend
- **Default**: 2 blocks (balanced precision/completeness)

### **Critical Bug Fixes**
- **Parent-Child Duplication**: Eliminated "echo chamber" content duplicates
- **Extraction Failure**: Fixed fragmented layout content loss
- **Threshold Optimization**: Lowered MIN_TEXT_LENGTH (150→100 chars)

---
*Generated for: X:\Aplikacje\topical-focus-analyzer\multi_sitemap_app.py*  
*Updated: August 1, 2025*  
*Total Lines: ~2100+ | Functions: 5+ | Session Variables: 26 | Tabs: 7*  
*Latest Enhancement: Enhanced Multi-Stage Content Extraction Engine V3*