from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from dotenv import load_dotenv
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destination(BaseModel):
    city: str = Field(description="A city that matches the user's interest")
    reason: str = Field(description="A brief explanation of why this city is a good match for the user's interest")


parser = JsonOutputParser(pydantic_object=Destination)

prompt_template = PromptTemplate(
    template="""
        Sugira uma cidade dado o meu interesse por {interest}.
        {format_instructions}
    """,
    input_variables=["interest"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)

chain = prompt_template | model | parser


response = chain.invoke({
    "interest": "praia"
})

print(response)