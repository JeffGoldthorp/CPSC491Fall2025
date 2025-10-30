# CPSC 491 Fall 2025 - Emergency Alert System Project

## Overview
This project implements an intelligent chatbot system for emergency alerting systems, focusing on the Emergency Alert System (EAS), Wireless Emergency Alerts (WEA), and Integrated Public Alert and Warning System (IPAWS).

## Features

### ChromaChat - Vector Database Chat System
- **Semantic Search**: Uses ChromaDB to store and retrieve relevant emergency alert documentation
- **Relevance Filtering**: Employs cosine similarity to ensure questions are related to emergency alerting systems (threshold: 0.35)
- **External Search Integration**: Automatically fetches additional information via SerpAPI when local knowledge is insufficient
- **Embedding Management**: Tracks and displays the number of embeddings in the database
- **Source Citations**: Provides responses with proper citations and sources

### Model Fine-Tuning
- Fine-tuned GPT model specifically for emergency alerting system queries
- Comparison tools to evaluate model performance against base GPT-4o-mini
- Training data improvement scripts

## Project Structure

```
.
├── VectordB/              # Vector database implementation
│   ├── ChromaChat.py      # Main chat interface with ChromaDB
│   ├── ChromaDB.py        # Database utilities
│   └── pinecone_chat.py   # Pinecone alternative implementation
├── Front-End/             # Frontend components
│   ├── Sequential_finetuning.py
│   └── create_jsonl.py
├── doc/                   # Documentation and text files
├── archive/               # Archived code and experiments
├── compare_models.py      # Model comparison and evaluation
├── config.py              # Configuration management
├── improve_training_data.py  # Training data enhancement
├── IMPROVEMENT_GUIDE.md   # Detailed guide for improving model performance
└── requirements.txt       # Python dependencies

```

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- SerpAPI key (optional, for external search)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JeffGoldthorp/CPSC491Fall2025.git
cd CPSC491Fall2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
SERPAPI_API_KEY=your_serpapi_key_here
CHROMA_PERSIST_PATH=./chroma_fcc_storage
CHROMA_COLLECTION=fcc_documents
```

## Usage

### Running ChromaChat

```bash
python3 VectordB/ChromaChat.py
```

The chat interface will:
- Display the current number of embeddings at startup
- Filter questions to ensure they're relevant to emergency alerting systems
- Automatically fetch and store external sources when needed
- Provide detailed responses with source citations

### Model Comparison

```bash
python3 compare_models.py
```

This will compare the fine-tuned model against the base GPT-4o-mini model across multiple metrics:
- Relevance to emergency alerting systems
- Question directness
- Specificity (names, dates, numbers, citations)
- Source citation frequency
- Hallucination risk
- Response length

## Key Features Explained

### Cosine Similarity Filtering
The system uses cosine similarity to determine if a user's question is relevant to emergency alerting systems. Questions must achieve a similarity score of at least 0.35 with predefined emergency topics to be answered.

**Emergency Topics Covered:**
- Emergency Alert System (EAS)
- Wireless Emergency Alerts (WEA)
- Integrated Public Alert and Warning System (IPAWS)
- FCC regulations and public safety communications
- Emergency management and disaster response
- Cybersecurity policy for emergency systems
- Critical infrastructure protection

### Embedding Management
- Displays total embeddings at startup
- Shows count of new embeddings added from external searches
- Maintains persistent storage in ChromaDB

### External Search Integration
When the local database lacks sufficient information, the system:
1. Searches Google via SerpAPI
2. Fetches full text content from relevant sources
3. Chunks and embeds the content
4. Stores embeddings in ChromaDB for future queries
5. Reports the number of new embeddings added

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for embeddings and chat completions
- `SERPAPI_API_KEY`: Optional, enables external search
- `CHROMA_PERSIST_PATH`: Path to ChromaDB storage (default: `./chroma_fcc_storage`)
- `CHROMA_COLLECTION`: Collection name (default: `fcc_documents`)

### Tunable Parameters (in ChromaChat.py)
- `RELEVANCE_THRESHOLD`: Cosine similarity threshold for topic relevance (default: 0.35)
- `SIMILARITY_TOP_K`: Number of similar documents to retrieve (default: 5)
- `MAX_RESPONSE_TOKENS`: Maximum tokens in response (default: 500)
- `CHUNK_SIZE`: Text chunk size for embedding (default: 2000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Training Data Improvement

See [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) for detailed instructions on:
- Enriching training data with more specificity
- Adding source citations
- Increasing emergency alerting terminology
- System prompt optimization
- Data augmentation strategies

## Contributing

This is a course project for CPSC 491 Fall 2025. 

## License

This project is for educational purposes as part of CPSC 491 coursework.

## Contact

For questions or issues, please contact the repository owner.
