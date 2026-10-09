#  PaperLens — Research Paper Explainer Assistant

**Team Name:** Smart Work

PaperLens is a Generative AI-powered research paper assistant that enables users to upload research papers in PDF format and interact with them through natural-language questions.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant passages from an uploaded paper and generate context-based answers. It also displays page-level source references and retrieved passages so users can verify the information.

PaperLens supports beginner-friendly and technical explanations, automatic research paper summarization, and multiple paper conversations through a Streamlit web interface.

## Team Members

| S. No. | Name | Register Number |
|---|---|---|
| 1 | M. Mariselvi | 99230041094 |
| 2 | S. Kavya | 99230041093 |
| 3 | Sampathi Yaswanth | 99230041039 |
| 4 | C. Manasa | 99230041047 |
| 5 | M. Ganesh | 99230041020 |

## Features

- Upload research papers in PDF format
- Extract text page by page using `pypdf`
- Split documents into overlapping text chunks
- Generate semantic embeddings using Sentence Transformers
- Store and retrieve document chunks using ChromaDB
- Ask natural-language questions about research papers
- Retrieve relevant content using semantic similarity
- Generate answers using Retrieval-Augmented Generation (RAG)
- Use Groq as the primary AI provider when configured
- Automatically fall back to Google Gemini when Groq fails
- Support Simple and Technical explanation modes
- Generate structured research paper summaries
- Display source page numbers and retrieved passages
- Manage multiple research paper conversations
- View recent chats, pin chats, clear messages, and delete chats
- Handle AI provider errors and model quota limitations

## System Architecture

```text
                 Research Paper PDF
                         |
                         v
                 PDF Text Extraction
                         |
                         v
                 Page-aware Chunking
                         |
                         v
               Sentence Transformer
                   Embeddings
                         |
                         v
                     ChromaDB
                  Vector Database
                         |
                 +-------+-------+
                 |               |
                 v               v
           User Question     Paper Summary
                 |               |
                 v               v
         Question Embedding   Paper Text
                 |
                 v
          Semantic Retrieval
                 |
                 v
         Relevant Paper Chunks
                 |
                 v
           Groq AI Provider
                 |
          If Groq fails
                 |
                 v
          Google Gemini
          Fallback Provider
                 |
                 v
          Generated Answer
                 |
                 v
       Answer + Source Pages
```

**Note:** Groq is the preferred provider when a supported model is available. Gemini is used as a fallback. If Groq is not configured, PaperLens can use Gemini directly.

## How PaperLens Works

### 1. PDF Upload

The user uploads a research paper through the Streamlit interface.

### 2. Text Extraction

`pypdf` extracts readable text from individual PDF pages. The original page numbers are retained for source references.

### 3. Text Chunking

The extracted text is split into smaller, overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

Current configuration:

| Parameter | Value |
|---|---|
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |

### 4. Embedding Generation

Each text chunk is converted into a numerical vector using the Sentence Transformers model:

`all-MiniLM-L6-v2`

### 5. Vector Storage

ChromaDB stores the text chunks, embeddings, and page-number metadata for retrieval.

### 6. Question Processing

When a user asks a question, PaperLens converts it into an embedding using the same embedding model.

### 7. Semantic Retrieval

ChromaDB retrieves the most relevant text chunks by comparing the question embedding with the stored document embeddings.

The current implementation retrieves up to five relevant chunks for question answering.

### 8. Answer Generation

The retrieved content is supplied to the configured AI provider along with instructions to answer using the research paper content.

- **Primary provider:** Groq, when configured and available
- **Fallback provider:** Google Gemini

The system attempts the Gemini models configured in the application if the Groq request fails.

### 9. Source Evidence

PaperLens displays retrieved passages and their corresponding page numbers to help users verify the generated response.

**Important:** Source passages indicate the retrieved evidence. They do not guarantee that every generated statement is fully supported by the paper.

## Explanation Modes

### Simple Mode

Provides beginner-friendly answers using simple English and minimal technical terminology.

Suitable for students who want to understand research concepts easily.

### Technical Mode

Provides detailed explanations using research terminology and appropriate computer science concepts.

Suitable for technical discussions, project reviews, and research analysis.

## Automatic Paper Summary

PaperLens generates a structured summary with the following sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms and Technologies Used
6. Main Results
7. Limitations
8. Conclusion

When the required information is absent from the supplied paper text, the model is instructed to report:

`Not specified in the paper.`

The summary is generated from the extracted paper text and is subject to the AI model's context and output limits.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Processing | pypdf |
| Text Splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Primary Generative AI | Groq |
| Fallback Generative AI | Google Gemini |
| Retrieval Method | Semantic Similarity |
| Architecture | Retrieval-Augmented Generation |
| Development Environment | Google Colab |
| Version Control | Git and GitHub |
| Deployment Option | Streamlit Community Cloud |

## Project Structure

The following is a suggested repository structure. Include only files and folders that actually exist in your repository.

```text
research_paper_explainer-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   └── research_paper_explainer.ipynb
│
├── screenshots/
│
└── .streamlit/
    └── secrets.toml  # Keep private; do not commit
```

The notebook can be developed in Google Colab. The Streamlit application is run using `app.py` in an environment where the required dependencies are installed.

## Installation and Setup

### Prerequisites

- Python 3.10 or another Python version supported by the installed dependencies
- A Groq API key for the primary provider, if using Groq
- A Google Gemini API key for the fallback provider
- Git, if cloning the repository

### 1. Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <your-github-repository-url>
cd research_paper_explainer-assistant
```

### 2. Install Dependencies

Install the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

For local execution, create the file:

`.streamlit/secrets.toml`

Add the API keys you have configured:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

If you do not have a Groq key, you can omit `GROQ_API_KEY` and use Gemini directly.

Never commit real API keys or `secrets.toml` to GitHub.

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Streamlit will display a local URL that you can open in your browser.

## API Configuration

PaperLens supports two AI providers.

| Provider | Purpose |
|---|---|
| Groq | Primary provider when configured and a supported model is available |
| Google Gemini | Fallback provider, or the main provider when Groq is unavailable |

### Groq Configuration

Create an API key through the [Groq Console](https://console.groq.com/keys).

### Gemini Configuration

Create a Gemini API key through [Google AI Studio](https://aistudio.google.com/apikey).

For Streamlit Community Cloud, configure the keys through the app's **Settings → Secrets** interface.

The application must have the corresponding API keys configured before those providers can be used. Provider availability, supported models, and API quotas can change.

## Example Questions

Users can ask questions such as:

- What problem does this paper solve?
- What methodology is used in this research?
- What dataset was used?
- Which algorithms are used?
- What are the main results?
- What are the limitations of this research?
- Explain the proposed approach in simple terms.
- How does the proposed model work?
- What are the contributions of this paper?
- Summarize the research paper.

## Advantages

- Reduces the time needed to understand research papers
- Supports natural-language interaction with academic documents
- Uses semantic retrieval to find relevant passages
- Provides page-level source references
- Supports beginner-friendly and technical explanations
- Generates structured summaries
- Supports multiple AI providers
- Can automatically try Gemini when Groq fails

## Limitations

- Extraction quality depends on the PDF's text structure.
- Scanned or image-only PDFs may require OCR.
- Semantic retrieval may miss relevant passages.
- Generated answers may contain unsupported interpretations despite source-grounding instructions.
- Page references identify retrieved passages, not necessarily every source for every claim.
- Large papers may exceed practical context or processing limits.
- AI responses depend on provider availability, supported models, and API quotas.
- The current in-memory ChromaDB client does not provide durable vector storage across application restarts.
- Chat history stored in Streamlit session state may not persist across sessions.

## Future Enhancements

- OCR support for scanned research papers
- Persistent vector database storage
- Persistent chat history and user accounts
- Multi-paper comparison
- More precise claim-level citations
- Improved retrieval using reranking and hybrid search
- Automatic figure and table understanding
- Export summaries as PDF
- Support for additional document formats
- Multilingual paper explanations
- More robust evaluation of answer relevance and factual correctness

## Deployment

PaperLens can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

General deployment steps:

1. Push the application source code to GitHub.
2. Create a Streamlit Community Cloud app connected to the repository.
3. Select the correct entry point, such as `app.py`.
4. Add the required API keys under the app's Secrets settings.
5. Deploy and test PDF upload, retrieval, summarization, and question answering.

Do not upload API keys, private research papers, or `secrets.toml` to a public repository.

## Project Information

**Project Title:** PaperLens — Research Paper Explainer Assistant  
**Team Name:** Smart Work  
**Domain:** Generative AI and Natural Language Processing  
**Core Approach:** Retrieval-Augmented Generation (RAG)  
**Interface:** Streamlit

PaperLens aims to make academic research easier to explore by combining semantic retrieval, generative AI, and source-page evidence in an interactive research assistant.
