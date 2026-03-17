import asyncio
import os

from dotenv import load_dotenv
from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END


class Route(TypedDict):
    destino: Literal["praia", "montanha"]

class State(TypedDict):
    query: str
    destination: Route
    response: str


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

async def router_node(state: State, config=RunnableConfig):
    return {"destination": await router.ainvoke({"query": state["query"]}, config)}


async def beach_node(state: State, config=RunnableConfig):
    return {"response": await beach_chain.ainvoke({"query": state["query"]}, config)}


async def mountain_node(state: State, config=RunnableConfig):
    return {"response": await mountain_chain.ainvoke({"query": state["query"]}, config)}


async def pick_node(state: State)-> Literal["praia", "montanha"]:
    return "praia" if state["destination"]["destino"] == "praia" else "montanha"


graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("praia", beach_node)
graph.add_node("montanha", mountain_node)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", pick_node)
graph.add_edge("praia", END)
graph.add_edge("montanha", END)

app = graph.compile()

async def main():
    response = await app.ainvoke({"query": "Quero escalar montanhas no sul do brasil. O que você sugere?"})
    print(response['response'])


asyncio.run(main())