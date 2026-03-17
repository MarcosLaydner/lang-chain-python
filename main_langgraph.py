from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict


class Route(TypedDict):
    destino: Literal["praia", "montanha"]

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    api_key=api_key
)

prompt_beach_consulting = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia. Responda brevemente e de forma objetiva."),
        ("human", "{query}")
    ]
)

prompt_mountain_consulting = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha. Você é um especialista em viagens com destinos para montanhas e atividades radicais. Responda brevemente e de forma objetiva"),
        ("human", "{query}")
    ]
)

beach_chain = prompt_beach_consulting | model | StrOutputParser()
mountain_chain = prompt_mountain_consulting | model | StrOutputParser()

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

router = router_prompt | model.with_structured_output(Route)

def response(query: str):
    route = router.invoke({"query": query})["destino"]
    if route == "praia":
        return beach_chain.invoke({"query": query})
    else:
        return mountain_chain.invoke({"query": query})


print(response("Quero me aventurar em lugar alto"))