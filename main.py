from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

days = 7
children = 2
interest = "praia"

prompt_template = PromptTemplate(
    template="""
        Sugira uma cidade dado o meu interesse por {interest}
    """,
    input_variables=["interest"],
)

model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)

chain = prompt_template | model | StrOutputParser()


response = chain.invoke({
    "interest": interest
})

print(response)