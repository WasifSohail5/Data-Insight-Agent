from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import ChatModel
import os
from dotenv import load_dotenv

load_dotenv()

def load_gemini_llm(model:str="models/chat-bison-1.5-flash") -> ChatModel:
    return ChatGoogleGenerativeAI(model=model,temperature=0.3,convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"))

