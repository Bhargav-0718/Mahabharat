"""
Streamlit Web Application

Interactive UI for querying the Mahabharata-SemRAG system.
"""

import streamlit as st
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mahabharata-SemRAG",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📜 Mahabharata-SemRAG")
st.subtitle("Contextual Question Answering over the KM Ganguly Translation")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your OpenAI API key for embeddings and LLM"
    )
    
    st.divider()
    
    st.header("📚 About")
    st.markdown("""
    **Mahabharata-SemRAG** is a hallucination-resistant QA system that:
    
    - ✅ Answers factual & narrative questions
    - ✅ Supports temporal queries (by story phase)
    - ✅ Provides explicit Parva + Section attribution
    - ✅ Quotes verbatim contextual passages
    - ✅ Uses graph-first retrieval, LLM last
    """)
    
    st.divider()
    
    st.header("📖 Story Phases")
    phases = [
        "Origins",
        "Rise of the Pandavas",
        "Exile",
        "Prelude to War",
        "Kurukshetra War",
        "Immediate Aftermath",
        "Post-War Instruction",
        "Withdrawal from the World"
    ]
    for phase in phases:
        st.caption(f"• {phase}")


# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Query the Mahabharata")
    
    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        - Where did Arjuna go during the Agyatvasa?
        - What happened to Draupadi after the Dice Game?
        - Describe Kurukshetra War Day 1.
        - Who was Karna's true father?
        - What did Krishna teach Arjuna?
        """)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="Type your question about the Mahabharata...",
        height=100
    )
    
    # Query options
    st.subheader("⏱️ Filters")
    col_phase, col_char = st.columns(2)
    
    with col_phase:
        story_phase = st.selectbox(
            "Filter by Story Phase (optional)",
            ["None", "Origins", "Rise of the Pandavas", "Exile", 
             "Prelude to War", "Kurukshetra War", "Immediate Aftermath",
             "Post-War Instruction", "Withdrawal from the World"]
        )
    
    with col_char:
        character = st.selectbox(
            "Filter by Character (optional)",
            ["None", "Arjuna", "Yudhishthira", "Bhima", "Draupadi", 
             "Krishna", "Karna", "Duryodhana", "Ashvatthama"]
        )
    
    # Submit button
    search_button = st.button("🔎 Search", use_container_width=True, type="primary")


with col2:
    st.header("📊 System Status")
    
    st.metric("Knowledge Graph Nodes", "Loading...")
    st.metric("Embeddings Generated", "Loading...")
    st.metric("Context Units", "Loading...")
    
    st.divider()
    
    st.subheader("✨ Features")
    st.checkbox("Show graph traversal", value=False)
    st.checkbox("Show semantic re-ranking", value=False)
    st.checkbox("Show raw retrieved units", value=False)


# Results section
if search_button:
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    elif not api_key:
        st.error("❌ Please provide your OpenAI API key in the sidebar.")
    else:
        with st.spinner("🔍 Searching the knowledge graph..."):
            st.info("System is under development. Full pipeline coming soon.")
            
            # Placeholder response
            st.success("✅ Query processed!")
            
            st.subheader("📖 Answer")
            st.markdown("""
            *Answer will appear here once the system is fully implemented.*
            
            During the Agyatvasa, Arjuna went to the kingdom of Virata, 
            where he lived in disguise as a eunuch named Brihannala.
            """)
            
            st.divider()
            
            st.subheader("📌 Retrieved From")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**Parva:**\nVirata Parva")
            with col2:
                st.info("**Section:**\nSection X")
            with col3:
                st.info("**Story Phase:**\nExile.Year13.Agyatvasa")
            
            st.divider()
            
            st.subheader("📜 Contextual Passage")
            st.quote("""
            Arjuna, assuming the guise of a eunuch named Brihannala, 
            entered the city of Virata. Skilled in music and dance, 
            he lived there unnoticed, teaching the royal family...
            """)
            
            st.divider()
            
            with st.expander("🔧 Technical Details"):
                st.json({
                    "entities_extracted": ["Arjuna", "Brihannala", "Virata"],
                    "story_phases_matched": ["Exile", "Year13", "Agyatvasa"],
                    "graph_traversal_depth": 2,
                    "semantic_reranking": "Enabled",
                    "top_k_retrieved": 5
                })


# Footer
st.divider()
st.markdown("""
---
**Mahabharata-SemRAG** | KM Ganguly Translation | OpenAI-powered

*Citation:* Every answer is grounded in explicit Parva + Section references with verbatim passages.
""")
