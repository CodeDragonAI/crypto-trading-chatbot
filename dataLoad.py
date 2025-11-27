from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def dataLoad():
    file_path = r'C:\Users\bt984\Desktop\chatbot new\data\Trading_Concepts_Master.pdf'
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len,
        separators = ['\n\n', '\n', ' ', '']
    )

    split_documents = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = Chroma.from_documents(split_documents, embeddings)

    return db