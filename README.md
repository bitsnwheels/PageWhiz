# 🌐 PagWhiz

A **LangChain-powered Question-Answering Chatbot** that allows users to **enter any webpage URL** and then **ask questions based on the content extracted from that webpage**.  
It automatically fetches, processes, embeds, and stores webpage text to enable interactive, context-aware conversations.

---

## 🚀 Features

- 🔗 **URL-based Context** — Paste any valid webpage URL, and the chatbot will understand and analyze its content.  
- 🧠 **LangChain Integration** — Uses LangChain’s document loaders, text splitters, embeddings, and conversational retrieval.  
- 💬 **Conversational Memory** — The bot remembers chat history for better contextual answers.  
- ⚡ **Local or API-based Models** — Works with OpenAI, Hugging Face, or local sentence-transformer models.  
- 🧱 **Vector Database** — FAISS or Chroma for fast semantic retrieval.  
- 🖥️ **Frontend Ready** — Optional Gradio or Streamlit interface for a clean chat experience.

---

## 🏗️ Architecture Overview
      ┌──────────────────────────────┐
      │          User Input           │
      │ (Enter webpage URL + query)   │
      └──────────────┬───────────────┘
                     │
             [1] WebPageLoader
                     │
      ┌──────────────▼───────────────┐
      │    Text Extraction & Split    │
      │ (RecursiveCharacterTextSplitter)
      └──────────────┬───────────────┘
                     │
             [2] Embeddings (OpenAI/HF)
                     │
      ┌──────────────▼───────────────┐
      │      Vector Store (FAISS)     │
      │   + Similarity Search Index    │
      └──────────────┬───────────────┘
                     │
              [3] LangChain Retriever
                     │
      ┌──────────────▼───────────────┐
      │   Chat Model (LLM Interface)  │
      │  (OpenAI / Gemini / Local LLM)│
      └──────────────┬───────────────┘
                     │
            💬 Response to User

---

## 🧩 Tech Stack

- **Python 3.10+**
- **LangChain** — document loading, text processing, and chains  
- **FAISS / Chroma** — vector database  
- **OpenAI / Hugging Face / Local Models** — embeddings and LLMs  
- **BeautifulSoup4 / Requests** — webpage scraping  
- **Gradio / Streamlit** — simple frontend UI  

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/langchain-url-chatbot.git
cd langchain-url-chatbot

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt


OPENAI_API_KEY=your_openai_api_key
# or if you’re using Hugging Face embeddings:
HUGGINGFACEHUB_API_TOKEN=your_hf_token
python app.py
streamlit run app.py
# or
python gradio_app.py
Enter URL: https://en.wikipedia.org/wiki/Artificial_intelligence
User: What are the main subfields of AI?
Bot: The major subfields include machine learning, computer vision, natural language processing, robotics, and expert systems.

