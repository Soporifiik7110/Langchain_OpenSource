# Importation des bibliothèques nécessaires
import os
import yfinance as yfin
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, Tool

# Chargement des variables d'environnement
load_dotenv()
model_name = os.getenv('MODEL_4')
number_batch = os.getenv('BATCH')
layers_gpu = os.getenv('layers')
temperature = os.getenv("TEMPERATURE")
number_ctx = os.getenv('MODEL_N_CTX')


# Initialisation du modèle
llm = LlamaCpp(
    model_path=model_name,
    n_batch=number_batch,
    n_ctx=16000,
    temperature=temperature,
    n_gpu_layers=layers_gpu,
    verbose=True
)

def finance(query: str) -> str:
    """Replace query by the ticker of the stock like GOOGL for Google (for exemple)"""
    ticker = yfin.Ticker(query)
    info = ticker.info
    price = info.get('currentPrice')
    return f"The current stock price of {query} is at {price}"

# Création d'une instance de l'outil

# Ajout de l'outil à la liste des outils
tools = [Tool(
              name="finance",
              func=finance,  # Suppression de .run
              description="This is a tool to have information about a stock like \
                   like the stock price and the financial state of the company.\
                   you have to use the ticker the company like goolge is (GOOGL).")]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(tools, llm,
                        AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                        memory=memory,
                        verbose=True)

while True:
    request = input('Votre_question >')
    txt = agent.run(request)

    # Initialisation de l'agent avec les outils
    print(txt)