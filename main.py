from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

days = 7
children = 2
interests = "praia"

prompt_template = PromptTemplate(
    template="""
        Crie um roteiro de viagem de {days}
        dias para uma familia com {children} crianças,
        que gosta de {interests}.
        Seja breve em suas respostas, e forneça apenas o essencial para cada dia, sem detalhes adicionais.
    """
)

prompt = prompt_template.format(
    days=days, children=children, interests=interests
)

print("Prompt gerado: \n", prompt)

model = ChatOpenAI(model="gpt-5-nano", temperature=1, api_key=api_key)

response = model.invoke(prompt)

print(response.content)


# Directly with OPENAI

# client = OpenAI(api_key=api_key)

#  prompt = f"Crie um roteiro de viagem de {days} dias para uma familia com {children} crianças, que gosta de {interests}. Seja breve em suas respostas, e forneça apenas o essencial para cada dia, sem detalhes adicionais."
# # try gpt-5-mini (a bit better) OR gpt-5-nano (cheaper)
# resposta = client.chat.completions.create(
#     model="gpt-5-mini",
#     messages=[
#         {
#             "role": "system",
#             "content": "Você é um assistente de planejamento de viagens, que cria roteiros personalizados com base nas preferências do usuário."
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )


# print(resposta.choices[0].message.content)