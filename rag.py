from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)

embeddings = OpenAIEmbeddings()

files = [
    "documents/GTB_gold_Nov23.pdf",
    "documents/GTB_platinum_Nov23.pdf",
    "documents/GTB_standard_Nov23.pdf",
]

documents = sum([PyPDFLoader(file).load() for file in files], [])

pieces = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
).split_documents(documents)

data = FAISS.from_documents(pieces, embeddings).as_retriever(search_kwargs={"k": 2})

prompt_insurance_consulting = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

chain = prompt_insurance_consulting | model | StrOutputParser()

def respond(query: str):
    parts = data.invoke(query)
    context = "\n\n".join([part.page_content for part in parts])
    return chain.invoke({
        "query": query,
        "contexto": context
    })


print(respond("como devo proceder caso tenha um item roubado e caso tenha o cartão gold?"))
