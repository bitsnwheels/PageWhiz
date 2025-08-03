from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_answer_from_url(url: str, question: str) -> str:
    """
    Loads content from a URL, creates a RAG chain, and answers a question.
    """
    try:
        # Set a User-Agent header to identify your requests
        headers = {
            "User-Agent": "PageWhiz/1.0 (yourname@example.com)"
        }
        loader = WebBaseLoader(url, header_template=headers)
        docs = loader.load()

        if not docs:
            return "Could not load any content from the provided URL. Please check if the URL is correct and public."

        # Extract page content
        page_content = docs[0].page_content

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents([page_content])

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()

        # Define the prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
            Answer ONLY from the provided context.
            If the context is insufficient, just say you don't know.
            Keep the answer concise and to the point.

            Context: {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        # Define the LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Get the answer
        answer = rag_chain.invoke(question)
        return answer

    except Exception as e:
        return f"An error occurred: {e}. Please ensure the URL is valid and accessible."