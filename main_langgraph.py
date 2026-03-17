from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    api_key=api_key
)

prompt_consulting = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens"),
        ("human", "{query}")
    ]
)

assistant = prompt_consulting | model | StrOutputParser()

print(assistant.invoke({"query": "Quero férias em praias no Brasil?"}))