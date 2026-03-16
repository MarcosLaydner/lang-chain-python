from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)


class Destination(BaseModel):
    city: str = Field(description="A city that matches the user's interest")
    reason: str = Field(description="A brief explanation of why this city is a good match for the user's interest")


class Restaurants(BaseModel):
    city: str = Field(description="A city that matches the user's interest")
    restaurants: list[str] = Field(description="A list of restaurants in the city that match the user's interest")


destination_parser = JsonOutputParser(pydantic_object=Destination)
restaurants_parser = JsonOutputParser(pydantic_object=Restaurants)

city_prompt_template = PromptTemplate(
    template="""
        Sugira uma cidade dado o meu interesse por {interest}.
        {format_instructions}
    """,
    input_variables=["interest"],
    partial_variables={"format_instructions": destination_parser.get_format_instructions()}
)

restaurants_prompt_template = PromptTemplate(
    template="""
        Sugira uma lista de restaurantes em {city}.
        {format_instructions}
    """,
    partial_variables={"format_instructions": restaurants_parser.get_format_instructions()}
)

cultural_prompt_template = PromptTemplate(
    template="""
        Sugira uma atividade cultural em {city}.
    """
)

chain_1 = city_prompt_template | model | destination_parser
chain_2 = restaurants_prompt_template | model | restaurants_parser
chain_3 = cultural_prompt_template | model | StrOutputParser()

chain = (chain_1 | chain_2 | chain_3)

response = chain.invoke({
    "interest": "praia"
})

print(response)