import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)

prompt_suggestion = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])

chain = prompt_suggestion | model | StrOutputParser()

memory = {}
session = 'aula_langchain'

def hystory_per_session(session : str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
    return memory[session]

questions = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=hystory_per_session,
    input_messages_key="query",
    history_messages_key="historico"
)

for question in questions:
    response = chain_with_memory.invoke(
        {
            "query": question,
        },
        config={"session_id": session}
    )
    print(f"Question: {question}")
    print(f"Response: {response}\n")