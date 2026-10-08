# PaperLens — Research Paper Explainer Assistant

**Team Name:** Smart Work

## Team Members

| S. No. | Name              | Register Number |
| ------ | ----------------- | --------------- |
| 1      | M. Mariselvi      | 99230041094     |
| 2      | S. Kavya          | 99230041093     |
| 3      | Sampathi Yaswanth | 99230041039     |
| 4      | C. Manasa         | 99230041047     |
| 5      | M. Ganesh         | 99230041020     |


PaperLens is a Generative AI-powered research paper assistant that allows users to upload a research paper in PDF format and interact with it through natural-language questions.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant sections from the uploaded paper and generate answers using Google Gemini. It also provides page-level source evidence so users can verify where an answer comes from.

## Features

* Upload research papers in PDF format
* Extract text from research papers page by page
* Split documents into meaningful text chunks
* Generate semantic embeddings using Sentence Transformers
* Store and retrieve document chunks using ChromaDB
* Ask natural-language questions about the paper
* Retrieval-Augmented Generation (RAG)
* Google Gemini-powered answers
* Simple explanation mode for beginners
* Technical explanation mode for detailed answers
* Automatic research paper summary
* Page-level source references
* Display retrieved source passages
* Multiple research-paper conversations
* Recent chats
* Pinned chats
* Gemini model fallback handling for quota limits

## System Architecture

```text
                    Research Paper PDF
                           │
                           ▼
                  PDF Text Extraction
                           │
                           ▼
                  Page-aware Chunking
                           │
                           ▼
                Sentence Transformer
                     Embeddings
                           │
                           ▼
                      ChromaDB
                  Vector Database
                           │
             ┌─────────────┴─────────────┐
             │                           │
        User Question              Paper Summary
             │                           │
             ▼                           ▼
      Question Embedding          Gemini Analysis
             │
             ▼
       Similarity Search
             │
             ▼
      Relevant Paper Chunks
             │
             ▼
        Gemini RAG Model
             │
             ▼
      Answer + Page Sources
```

## How PaperLens Works

### 1. PDF Upload

The user uploads a research paper in PDF format.

### 2. Text Extraction

PaperLens extracts readable text from every page using `pypdf`.

### 3. Text Chunking

The extracted content is divided into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk size: 1000
Chunk overlap: 200
```

### 4. Embedding Generation

Each text chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### 5. Vector Storage

The generated embeddings and corresponding text are stored in ChromaDB.

Page numbers are stored as metadata so that the retrieved information can be traced back to the original paper.

### 6. Question Processing

When the user asks a question, the question is converted into an embedding using the same embedding model.

### 7. Similarity Retrieval

ChromaDB retrieves the most relevant document chunks based on semantic similarity.

### 8. RAG Generation

The retrieved paper content is provided as context to Google Gemini.

The model is instructed to answer using only the retrieved research-paper content.

### 9. Source Evidence

PaperLens displays the retrieved passages and their page numbers, allowing users to verify the generated answer.

## Explanation Modes

### Simple Mode

Provides beginner-friendly explanations using simple English and minimal technical terminology.

### Technical Mode

Provides more detailed explanations using appropriate research and computer science terminology.

## Paper Summary

PaperLens can automatically generate a structured summary containing:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms and Technologies Used
6. Main Results
7. Limitations
8. Conclusion

If information is not available in the paper, the system reports:

```text
Not specified in the paper.
```

## Technologies Used

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| User Interface       | Streamlit                      |
| PDF Processing       | pypdf                          |
| Text Splitting       | LangChain                      |
| Embeddings           | Sentence Transformers          |
| Embedding Model      | all-MiniLM-L6-v2               |
| Vector Database      | ChromaDB                       |
| Generative AI        | Google Gemini                  |
| Retrieval Method     | Semantic Similarity            |
| Architecture         | Retrieval-Augmented Generation |

## Project Structure

```text
research_paper_explainer-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│
├── notebooks/
│   └── research_paper_explainer.ipynb
│
├── src/
│
└── screenshots/
```

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd research_paper_explainer-assistant
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## API Key Configuration

PaperLens requires a Google Gemini API key.

For local development, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

For Streamlit Community Cloud, add the same key through the application's Secrets settings.

Do not upload API keys or `secrets.toml` to GitHub.

## Example Questions

Users can ask questions such as:

```text
What methodology is used in this paper?

What dataset was used?

What algorithms are used?

What are the main results?

What are the limitations?

Explain the proposed approach in simple terms.

How does the proposed model work?
```

## Advantages

* Reduces the time required to understand research papers
* Enables natural-language interaction with academic documents
* Provides evidence for generated answers
* Supports both beginner-friendly and technical explanations
* Uses semantic retrieval rather than simple keyword matching
* Helps users quickly identify methodology, datasets, results, and limitations

## Limitations

* Performance depends on the quality of extracted PDF text
* Scanned/image-only PDFs may require OCR
* The current vector database is maintained in memory
* Very large research papers may require additional document management
* Generated answers depend on the quality of retrieved context
* Gemini API usage is subject to model/API quota limits

## Future Enhancements

* OCR support for scanned research papers
* Persistent vector database
* Multi-paper comparison
* Citation-aware answers
* Automatic figure and table understanding
* Research-paper recommendation
* Export summaries as PDF
* Authentication and user accounts
* Persistent chat history
* Support for additional document formats

## Deployment

The application can be deployed using Streamlit Community Cloud.

The Gemini API key should be stored securely using Streamlit Secrets rather than inside the source code.

## Project

**PaperLens — Research Paper Explainer Assistant**

A Generative AI and Retrieval-Augmented Generation application for interactive research paper understanding.
