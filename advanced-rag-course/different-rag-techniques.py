import json

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.evaluation.qa import QAEvalChain, QAGenerateChain

from transformers import AutoTokenizer

load_dotenv()

pdfs = DirectoryLoader("shared_documents", glob="*.pdf").load()

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, chunk_size=1250, chunk_overlap=150
)

chunks = splitter.split_documents(pdfs)

embeddings = OllamaEmbeddings(model="bge-m3:567m")

vector_store = FAISS.from_documents(
    documents=chunks, embedding=embeddings
)

retriever = vector_store.as_retriever()

model = OllamaLLM(model="gemma3:4b")
gpt = ChatOpenAI(model="gpt-5-nano")


prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Responda usando exclusivamente os conteúdo fornecido. \n\nContexto:\n{contexto}"),
            ("human", "{query}")
        ]
    )

query = "como devo proceder caso tenha um item roubado e caso tenha o cartão gold?"


def rag_chain():
    return (
        {
            "contexto": RunnablePassthrough() | retriever,
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

# Query Chain using a rewritter to improve the search in the vector database
def rewriter_rag_chain():
    rewriter_prompt_template = PromptTemplate.from_template(
        """
            Gere consulta de pesquisa para o banco de dados de vetores (Vector DB) a partir de uma pergunta do usuário,
            permitindo uma resposta mais precisa por meio da busca semantica.
            Basta retornar a consulta revisada do Vector DB, entre aspas.

            Pergunta do usuário: {user_question}

            Consulta revisada do Vector DB:
        """
    )

    rewriter_chain = rewriter_prompt_template | model | StrOutputParser()

    return (
        {
            "contexto": RunnablePassthrough() | rewriter_chain | retriever,
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

# query chain using multiquery retriever to search in the vector database with multiple queries
def multi_query_rag_chain():
    multi_query_prompt_template = PromptTemplate.from_template(
        """
            Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
            versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
            Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
            o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
            Forneça estas perguntas alternativas separadas por quebras de linha.
            Pergunta original: {question}
        """
    )

    multi_query_chain = multi_query_prompt_template | gpt | CommaSeparatedListOutputParser()

    multi_retriever = MultiQueryRetriever(retriever=retriever, llm_chain=multi_query_chain)

    return (
        {
            "contexto": RunnablePassthrough() | multi_query_chain | multi_retriever,
            "query": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

# query chain using HyDE (Hypothetical Document Embeddings) to generate um documento hipotético a partir da pergunta do usuário e usar esse documento para buscar no banco de dados vetorial
def hyde_rag_chain():
    hyde_chain = gpt | StrOutputParser()

    return (
        {
            "contexto": RunnablePassthrough() | hyde_chain | retriever,
            "query": RunnablePassthrough()
        }
        | prompt
        | gpt
        | StrOutputParser()
    )

# Print different outputs to compare the differnt rag approaches

# print("RAG Chain:")
# print(rag_chain().invoke(query))
# print("------------------------------")
# print("RAG Chain with Rewriter:")
# print(rewriter_rag_chain().invoke(query))
# print("------------------------------")
# print("RAG Chain with MultiQuery Retriever:")
# print(multi_query_rag_chain().invoke(query))
# print("------------------------------")
# print("RAG Chain with HyDE:")
# print(hyde_rag_chain().invoke(query))

eval_chain = QAEvalChain.from_llm(gpt)

def evaluate(query_answers, generations):
    # query_answers: query, answer
    # generations: result
    evaluations = eval_chain.evaluate(query_answers, generations)
    correct_results = 0
    for i in enumerate(query_answers):
        correct_results = correct_results + (1 if evaluations[i[0]]["results"].split("\n")[-1].split(":")[-1].strip() == "CORRECT" else 0)
    return correct_results/len(query_answers)
        

# qa_chain = QAGenerateChain.from_llm(gpt)

# query_answers = qa_chain.apply_and_parse(
#     [ {"doc": p.page_content } for p in chunks ]
# )

with open("test_qa.json", "r") as f:
    question_answers = json.load(f)
