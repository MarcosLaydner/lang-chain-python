import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


document = TextLoader("shared_documents/GTB_gold_Nov23.txt").load()[0]

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
).split_documents([document])

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=api_key)

vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks, embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "como devo proceder caso tenha um item roubado e caso tenha o cartão gold?"

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente os conteudo fornecido. \n\nContexto:\n{contexto}"),
    ("human", "{query}")
])

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=1,
    api_key=api_key
)

chain = prompt | model | StrOutputParser()

retrieved_chunks = retriever.invoke(query)
context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

print(chain.invoke({"query": query, "contexto": context}))
