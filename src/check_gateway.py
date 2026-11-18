"""Check that the gateway answers, for the chat model and for the embedder."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
llm = ChatOpenAI(model=os.environ["CHAT_MODEL"], base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
print("Modèle de chat :", llm.invoke("Répondez seulement : prêt.").content)

embeddings = OpenAIEmbeddings(model=os.environ["EMBED_MODEL"], base_url=os.environ["LLM_BASE_URL"],
                              api_key=os.environ["LLM_API_KEY"], check_embedding_ctx_length=False)
print("Embedder :", len(embeddings.embed_query("congé de maternité")), "nombres")
