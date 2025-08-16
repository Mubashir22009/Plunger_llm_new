from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import json

from pydantic import BaseModel
import uvicorn

## which model to use
#LLM = "Gemini"
LLM = "Groq"

load_dotenv(find_dotenv())  
# --- LLM Setup ---
if LLM == "Gemini":
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.0-flash",
        temperature=0
    )
elif LLM == "Groq":
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )

def get_chat_completion(prompt, model) -> str:
    response = model.invoke(prompt)
    return response.content.strip()

def llm_chat(user_query):
    prompt_injection = """You are a plunger lift engeenier working in a gas drill which employes use of the plunger lift to clear liquid/gunk buildup in tubing. You are given various sensor and other reading of the past day. Answer user's question by recalling the complete working of plunger lift system and taking into account the data given, to the best of your abilities. Keep in mind, you do not tell the user about columns as such, user doesn't know anything of the data. Note that if asked anything unrelated the the current system or plungers in general, apoligize and tell the user you can't answer this. Try to keep you messages brief and to the point"""

    well = "La Vista 1H"
    
    cycles = pd.read_csv(Path.cwd() / 'data' / 'cycles' / f'{well}_cycles.csv')
    # TODO: current day
    # Filter cycles where 'start_time' is on 2025-07-26
    # one day data
    cycles['__start_time'] = pd.to_datetime(cycles['start_time'])
    cycles = cycles[cycles['__start_time'].dt.date == pd.to_datetime('2025-07-26').date()]
    cycles.drop(columns=['__start_time'], inplace=True)
    cycles.reset_index(inplace=True)

    with open(Path.cwd() / 'data' / 'header_plot_descriptions.json') as f:
        descriptions = json.load(f)
        descriptions = {
        item["param_id"]: item["description"]["header_description"]
        for item in descriptions
    }

    prompt = f"""
    {prompt_injection}

    The following data is from the well named: {well}

    {cycles.to_string()}

    The descriptions for the parameters are as follows:
    {json.dumps(descriptions, indent=2)}

    Answer the following user query:
    {user_query}
    """

    return get_chat_completion(prompt, llm)

class ChatRequest(BaseModel):
    user_query: str

def main():
    fastapi = False
    args = sys.argv[1:]
    if "--fastapi" in args:
        fastapi = True

    if not fastapi:
        user_query = input("Enter your plunger lift question: ")
        response = llm_chat(user_query)
        print(response)
    else:
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Change to frontend URL in prod
            allow_methods=["POST"],
            allow_headers=["*"]
        )

        @app.post("/chat")
        def chat(req: ChatRequest):
            user_query = req.user_query
            response = llm_chat(user_query)

            return {
                "summary": response
            }

        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()