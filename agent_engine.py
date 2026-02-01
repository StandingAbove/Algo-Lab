import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import yfinance as yf

load_dotenv()

# 1. Define the State
class AgentState(TypedDict):
    ticker: str
    data: str
    report: str

# 2. Initialize Groq (Llama 3.3 70B is incredibly fast for agents)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. Nodes
def researcher(state: AgentState):
    """Fetch real-market data using yfinance."""
    ticker_obj = yf.Ticker(state['ticker'])
    hist = ticker_obj.history(period="5d")
    return {"data": hist.to_string()}

def analyst(state: AgentState):
    """Reasoning node using Groq."""
    prompt = f"""
    Analyze this stock data for {state['ticker']}:
    {state['data']}
    
    Provide a professional 3-bullet point quant summary focusing on:
    - Recent Price Trend
    - Volatility Observation
    - Key Technical Level
    """
    response = llm.invoke(prompt)
    return {"report": response.content}

# 4. Build the Agentic Loop
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

# Compile
research_app = workflow.compile()