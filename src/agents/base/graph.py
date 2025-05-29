from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from contextlib import asynccontextmanager
from agents.base.prompt import CALENDAR_AGENT_PROMPT
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def build_agent():

    today = datetime.now().strftime("%Y-%m-%d")

    graph = create_react_agent(
            model=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
            ),
            name="pirate_agent",
            prompt=CALENDAR_AGENT_PROMPT.render(today=today)
        )
        
    yield graph
