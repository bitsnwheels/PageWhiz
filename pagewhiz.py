from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator
from langdetect import detect
from bs4 import BeautifulSoup

def maybe_translate(text: str) -> str:
    """Detects language and translates a single text chunk to English if necessary."""
    try:
        # No need to sample, chunks are small enough
        lang = detect(text)
    except:
        # If detection fails, assume it's English or something went wrong
        return text

    if lang == "en":
        return text
    else:
        # Translating small chunks is much more reliable
        return GoogleTranslator(source='auto', target='en').translate(text)


def get_answer_from_url(url: str, question: str) -> str:
    """
    Loads content from a URL, creates a RAG chain, and answers a question.
    """
    try:
        headers = { "User-Agent": "PageWhiz/1.0 (yourname@example.com)" }
        loader = WebBaseLoader(url, header_template=headers)
        docs = loader.load()

        if not docs:
            return "Could not load any content from the provided URL."

        # Use BeautifulSoup to parse the HTML and extract only the main article text
        soup = BeautifulSoup(docs[0].page_content, "html.parser")
        content_div = soup.find(id="mw-content-text")
        
        if content_div:
            page_content = content_div.get_text(separator=" ", strip=True)
        else:
            page_content = soup.get_text(separator=" ", strip=True)

        # --- START: NEW ROBUST TRANSLATION LOGIC ---
        # 1. Split the cleaned text into manageable chunks *before* translation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_text(page_content)

        # 2. Translate each chunk individually. This is much more stable.
        translated_chunks = [maybe_translate(chunk) for chunk in text_chunks]
        # --- END: NEW ROBUST TRANSLATION LOGIC ---

        # 3. Create embeddings and vector store from the translated chunks
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Use .from_texts() which is designed for lists of strings
        vector_store = FAISS.from_texts(texts=translated_chunks, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # Define the prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
            Answer ONLY from the provided context.
            If the context is insufficient, just say you could not extract the asked information.
            Keep the answer concise and to the point.

            Context: {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        return answer

    except Exception as e:
        return f"An error occurred: {e}. Please ensure the URL is valid and accessible."
