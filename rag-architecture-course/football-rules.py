import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

PDF_PATH = "rag-architecture-course/documents/regras_futebol.pdf"
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# load the PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# create embeddings for the chunks
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# create a vectorstore from the chunks and their embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_regras_futebol"
)

# create a retriever from the vectorstore
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# create a language model for answering questions
llm = ChatOpenAI(
    model="gpt-5-nano",
    openai_api_key=api_key
)

# create a RetrievalQA chain that uses the retriever and the language model to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

pergunta = "Um jogador pode usar a mão para marcar um gol?"

resposta = qa_chain(pergunta)

print("Pergunta:")
print(pergunta)

print("\nResposta do Agente:")
print(resposta["result"])

print("\nTrechos utilizados como contexto:\n")

for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Fonte: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print("Conteúdo:")
    print(doc.page_content)
    print("\n")
