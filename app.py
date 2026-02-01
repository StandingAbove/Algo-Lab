import streamlit as st
from agent_engine import research_app

st.set_page_config(page_title="Groq Quant Agent", layout="wide")

st.title("⚡ Groq-Powered Quant Agent")
st.markdown("""
**System Status:** Running on **Groq LPU™** (Inference Speed: ~500 t/s)  
**Architecture:** LangGraph Multi-Node State Machine
""")

ticker = st.text_input("Enter Ticker Symbol", "NVDA")

if st.button("Execute Agentic Research"):
    with st.status("Agent traversing graph nodes...", expanded=True) as status:
        st.write("📡 Node: Researcher (Fetching yfinance data...)")
        
        # Invoke the LangGraph workflow
        initial_state = {"ticker": ticker, "data": "", "report": ""}
        result = research_app.invoke(initial_state)
        
        st.write("🧠 Node: Analyst (Groq Llama-3.3 reasoning...)")
        status.update(label="Workflow Complete!", state="complete")

    # Display Results
    st.divider()
    st.subheader(f"Quant Analysis: {ticker}")
    st.markdown(result["report"])
    
    with st.expander("Audit Trail: Raw Market Data"):
        st.text(result["data"])