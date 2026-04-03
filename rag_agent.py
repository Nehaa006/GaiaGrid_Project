import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load variables from .env (including your new GROQ_API_KEY)
load_dotenv()

class GaiaConsultant:
    def __init__(self, vector_db):
        # We use the key you just saved in your .env
        self.llm = ChatGroq(
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        self.vector_db = vector_db

    def explain_action(self, price, action, context_query):
        try:
            # 1. Search the vector database for relevant energy context
            # We look for the top 1 result (k=1)
            info = self.vector_db.similarity_search(context_query, k=1)
            
            # Check if we actually found something in the text files
            if info:
                context_text = info[0].page_content
            else:
                context_text = "Standard grid optimization protocols."

            # 2. Build the prompt for Llama 3
            prompt = f"""
            You are the Gaia-Grid Battery Consultant. 
            The battery just performed a {action} at a price of €{price:.2f}.
            
            Using this context: {context_text},
            explain to the user in 2 sentences why this was a smart financial move.
            """
            
            # 3. Get the explanation from Groq
            response = self.llm.invoke(prompt)
            
            # Return the text content
            return response.content
            
        except Exception as e:
            return f"Gaia is currently analyzing the market. (Offline Mode: {str(e)})"