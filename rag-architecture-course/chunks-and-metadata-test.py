import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

leaftlets = [
    "rag-architecture-course/documents/dipirona.pdf",
    "rag-architecture-course/documents/paracetamol.pdf",
]

documents = []

for leaflet in leaftlets:
    loader = PyPDFLoader(leaflet)
    docs = loader.load()

    for doc in docs:
        doc.metadata["medicamento"] = leaflet.split("/")[-1].replace(".pdf", "")
    
    documents.extend(docs)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,      # Tamanho máximo de cada chunk
    chunk_overlap=150   # Sobreposição entre chunks
)

chunks = text_splitter.split_documents(documents)

for chunk in chunks:

    # Normaliza o text para facilitar as verificações
    text = chunk.page_content.lower()

    # Identificação do medicamento
    if "identificação do medicamento" in text or "composição" in text:
        chunk.metadata["categoria"] = "identificacao"

    # Indicações terapêuticas
    elif "indicação" in text or "para que este medicamento é indicado" in text:
        chunk.metadata["categoria"] = "indicacao"

    # Funcionamento do medicamento
    elif "como este medicamento funciona" in text or "ação" in text:
        chunk.metadata["categoria"] = "como_funciona"

    # Contraindicações
    elif "contraindicação" in text or "quando não devo usar" in text:
        chunk.metadata["categoria"] = "contraindicacao"

    # Advertências e precauções
    elif "advertência" in text or "precaução" in text or "o que devo saber antes de usar" in text:
        chunk.metadata["categoria"] = "advertencias_precaucoes"

    # Interações medicamentosas
    elif "interação" in text or "interações medicamentosas" in text:
        chunk.metadata["categoria"] = "interacoes"

    # Posologia e modo de uso
    elif "dose" in text or "posologia" in text or "como devo usar" in text:
        chunk.metadata["categoria"] = "posologia_modo_uso"

    # Reações adversas
    elif "reações adversas" in text or "quais os males" in text:
        chunk.metadata["categoria"] = "reacoes_adversas"

    # Armazenamento
    elif "onde, como e por quanto tempo posso guardar" in text or "armazenar" in text:
        chunk.metadata["categoria"] = "armazenamento"

    # Superdosagem
    elif "quantidade maior do que a indicada" in text or "superdosagem" in text:
        chunk.metadata["categoria"] = "superdosagem"

    # Conteúdo geral / administrativo
    else:
        chunk.metadata["categoria"] = "geral"


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # opcional, mas recomendado
)

# Cria o banco vetorial com os chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_bulas"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Número de chunks retornados
)

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

pergunta = "Quais são as contraindicações da dipirona?"

# Executa a pergunta no agente RAG (forma atual)
resposta = qa_chain.invoke(pergunta)

print("Pergunta:")
print(pergunta)

print("\nResposta do Agente:")
print(resposta["result"])

print("\nTrechos utilizados como contexto:\n")

# Percorre os documentos recuperados
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")

    # Metadados principais
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")

    # Conteúdo recuperado
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n")
