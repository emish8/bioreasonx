import os
# Force pure-Python implementation for protobuf to resolve version mismatches in the environment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import requests
import time
import json
import plotly.graph_objects as go
from typing import Dict, Any

# Direct import fallbacks (makes app run without separate API process if needed)
try:
    from backend.services.llm_client import LLMClient
    from backend.services.data_fetcher import DataFetcher
    from backend.graph.knowledge_graph import BiomedicalKnowledgeGraph
    from backend.rag.vector_store import BiomedicalVectorStore
    from backend.agents.biomedical_agents import BiomedicalAgents
    from backend.workflows.reasoning_workflow import BioReasonWorkflow
    from backend.services.audio_processor import AudioProcessor
    DIRECT_BACKEND_AVAILABLE = True
except ImportError:
    DIRECT_BACKEND_AVAILABLE = False

# FastAPI Base URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="BioReason-X | Mutation-to-Therapy Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets/style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback raw style if path issue
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background-color: #0B0F19; color: #E2E8F0; }
        .bio-card { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; margin-bottom: 15px; }
        </style>
        """, unsafe_allow_html=True)

load_css()

# --- Initialize Backend Singletons for Direct Fallback Mode ---
@st.cache_resource
def get_backend():
    if DIRECT_BACKEND_AVAILABLE:
        llm = LLMClient()
        fetcher = DataFetcher()
        graph = BiomedicalKnowledgeGraph()
        vector_store = BiomedicalVectorStore()
        agents = BiomedicalAgents(llm, fetcher, graph, vector_store)
        workflow = BioReasonWorkflow(agents)
        audio = AudioProcessor()
        return workflow, graph, audio
    return None, None, None

workflow_direct, graph_direct, audio_direct = get_backend()

# --- Helper function to query backend ---
def run_reasoning_pipeline(mutation_query: str) -> Dict[str, Any]:
    # Try querying the FastAPI backend first
    try:
        response = requests.post(f"{API_URL}/reason", json={"query": mutation_query}, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("success"):
                # Fetch graph data from API
                graph_res = requests.get(f"{API_URL}/graph", timeout=5)
                graph_data = graph_res.json() if graph_res.status_code == 200 else {}
                return {"state": res_json["state"], "graph_source": "api", "graph_data": graph_data}
    except Exception:
        pass
        
    # Direct execution fallback (offline/in-process run)
    if workflow_direct:
        result_state = workflow_direct.execute(mutation_query)
        return {"state": result_state.model_dump(), "graph_source": "direct", "graph_data": None}
        
    return {}

# --- Header ---
st.markdown('<div class="glow-title">🧬 BioReason-X</div>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.15rem; color: #94A3B8; margin-top:-10px;">Explainable Biomedical Multi-Agent Reasoning Platform</p>', unsafe_allow_html=True)
st.markdown('---')

# --- Sidebar Inputs ---
st.sidebar.image("https://img.icons8.com/nolan/96/dna-helix.png", width=70)
st.sidebar.header("Mutation Ingestion")

# Presets & Input options
presets = ["BRCA1 c.5266dupC", "EGFR L858R", "BRAF V600E", "KRAS G12C", "ALK F1174L", "Custom"]
selected_preset = st.sidebar.selectbox("Select Target Mutation Profile", presets)

mutation_input = ""
if selected_preset == "Custom":
    mutation_input = st.sidebar.text_input("Enter Genomic Mutation (e.g. BRAF V600E)", "")
else:
    mutation_input = selected_preset

# Clinical audio recorder voice notes
st.sidebar.subheader("Clinical Dictation")
uploaded_audio = st.sidebar.file_uploader("Upload Clinician Audio Record", type=["mp3", "wav", "m4a"])

if uploaded_audio and selected_preset == "Custom":
    # Process audio notes using direct transcription helper
    if audio_direct:
        # Save temp file
        temp_audio_path = os.path.join("data", uploaded_audio.name)
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        transcribed_text = audio_direct.transcribe_clinician_note(temp_audio_path)
        st.sidebar.info(f"Transcribed note:\n*{transcribed_text}*")
        mutation_input = transcribed_text
        
        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass

run_analysis = st.sidebar.button("Run Causal Curation")

# --- Pipeline Execution ---
if run_analysis and mutation_input:
    # Reset/clear previous session
    if "result" in st.session_state:
        del st.session_state["result"]
        
    # Render Agent Progress Simulator
    progress_box = st.empty()
    bar = st.progress(0)
    
    agent_stages = [
        "Agent 1: Ingesting Variant from NCBI databases...",
        "Agent 2: Mapping Protein Domains & UniProt IDs...",
        "Agent 3: Resolving Reactome Cellular Pathways...",
        "Agent 4: Querying PubMed Literature & Vectorizing abstracts...",
        "Agent 5: Matching Target drugs from DGIdb...",
        "Agent 6: Auditing claims & generating confidence scores...",
        "Agent 7: Synthesizing Final Consensus Report..."
    ]
    
    for idx, stage in enumerate(agent_stages):
        progress_box.info(stage)
        bar.progress(int((idx + 1) / len(agent_stages) * 100))
        time.sleep(0.5) # Simulate execution transitions
        
    # Execute actual pipeline
    with st.spinner("Compiling Agent reports..."):
        result = run_reasoning_pipeline(mutation_input)
        if result:
            st.session_state["result"] = result
            st.success("Analysis Completed Successfully!")
            progress_box.empty()
            bar.empty()
        else:
            st.error("Error during agent reasoning pipeline execution. Please verify backend service.")

# --- Dashboard Display ---
if "result" in st.session_state:
    res = st.session_state["result"]
    state = res["state"]
    
    # Generate MP3 Audio file for report if Direct mode active
    audio_path = None
    if audio_direct and state.get("consensus", {}).get("final_consensus_statement"):
        audio_path = audio_direct.synthesize_report(state["consensus"]["final_consensus_statement"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Causal Curation Report", 
        "⚙️ Multi-Agent Trace", 
        "🕸️ Pathway Knowledge Graph", 
        "📚 Evidence Explorer"
    ])
    
    # --- Tab 1: Causal Curation Report ---
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="bio-card">', unsafe_allow_html=True)
            st.subheader("🧬 Genomic Variant Curation")
            ma = state.get("mutation_analysis") or {}
            st.markdown(f"**Gene Symbol:** `{ma.get('gene_symbol', 'N/A')}`")
            st.markdown(f"**Nomenclature:** `{ma.get('variant_nomenclature', 'N/A')}`")
            st.markdown(f"**Mutation Type:** `{ma.get('mutation_type', 'N/A')}`")
            st.markdown(f"**ClinVar Classification:** <span class='badge-pathogenic'>{ma.get('pathogenicity', 'N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"**Clinical Summary:** {ma.get('clinical_summary', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="bio-card">', unsafe_allow_html=True)
            st.subheader("🔬 Protein & Sequence Impact")
            gp = state.get("gene_protein") or {}
            st.markdown(f"**UniProt ID:** `{gp.get('uniprot_id', 'N/A')}`")
            st.markdown(f"**Protein Domains:** {gp.get('protein_domains', 'N/A')}")
            st.markdown(f"**Functional Consequence:** {gp.get('functional_consequence', 'N/A')}")
            st.markdown(f"**Molecular Mechanism:** {gp.get('molecular_mechanism', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="bio-card" style="border: 1px solid rgba(0, 242, 254, 0.3) !important;">', unsafe_allow_html=True)
            st.subheader("🎯 Therapeutic Recommendations")
            th = state.get("therapy") or {}
            val = state.get("validation") or {}
            
            st.markdown(f"**FDA-Approved/Investigational Therapies:**")
            for drug in th.get("recommended_therapies", []):
                st.markdown(f"- **`{drug}`**")
                
            st.markdown(f"**Mechanism of Action:** {th.get('mechanism_of_action', 'N/A')}")
            st.markdown(f"**Resistance / Escape Risks:** {th.get('resistance_risks', 'N/A')}")
            st.markdown(f"**Evidence Rating:** <span class='badge-mechanism'>{val.get('evidence_rating', 'N/A')}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="bio-card">', unsafe_allow_html=True)
            st.subheader("⚖️ Clinical consensus & Validation")
            cons = state.get("consensus") or {}
            st.markdown(f"**Consensus Confidence Rating:** `{cons.get('confidence_rating', 'N/A')}`")
            st.markdown(f"**Clinical Recommendations:** {cons.get('final_consensus_statement', 'N/A')}")
            
            # Actionable steps
            st.markdown("**Actionable Steps:**")
            for action in cons.get("recommended_actions", []):
                st.markdown(f"- {action}")
                
            # Audio Report
            if audio_path and os.path.exists(audio_path):
                st.markdown("🔊 **Vocal Summary Report:**")
                st.audio(audio_path, format="audio/mp3")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: Multi-Agent Trace ---
    with tab2:
        st.subheader("⚙️ LangGraph Chronological Agent Reasoning Trace")
        st.write("Below is the execution log of the 7 collaborating agents, showing the structured inference progression:")
        
        trace = state.get("reasoning_trace", [])
        for step in trace:
            st.markdown(f'<div class="agent-step completed">✔️ {step}</div>', unsafe_allow_html=True)

        st.markdown('<div class="bio-card">', unsafe_allow_html=True)
        st.subheader("🧠 Explainable Reasoning Progression Chain")
        st.markdown("""
        The agents traversed the causal reasoning chain in sequence:
        1. **Mutation Analysis Agent** identified variant nomenclatures and verified pathogenetic relevance.
        2. **Gene/Protein Agent** fetched matching structural domains and mapped consequences.
        3. **Pathway Agent** mapped the cell-cycle cascades affected.
        4. **Literature Agent** read PubMed research studies and indexed them in ChromaDB.
        5. **Therapy Agent** queried drug-gene targets.
        6. **Validation Agent** cross-referenced claims to establish validation ratings.
        7. **Consensus Agent** unified all inputs into the final report.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 3: Pathway Knowledge Graph ---
    with tab3:
        st.subheader("🕸️ Relational Causal Pathway Network Graph")
        st.write("Dynamic graph representing: **Mutation → Gene → Protein → Pathway → Disease → Drug**")
        
        # Load graph from API or Direct
        fig = None
        if res.get("graph_source") == "api" and res.get("graph_data"):
            # Build network figure from API graph nodes/links
            api_g = BiomedicalKnowledgeGraph()
            gd = res["graph_data"]
            for node in gd.get("nodes", []):
                api_g.graph.add_node(node["id"], type=node["type"], label=node["label"], color=node["color"])
            for link in gd.get("links", []):
                api_g.graph.add_edge(link["source"], link["target"], relation=link["relation"])
            fig = api_g.generate_plotly_figure()
        elif graph_direct:
            fig = graph_direct.generate_plotly_figure()
            
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active knowledge graph generated.")

    # --- Tab 4: Evidence Explorer ---
    with tab4:
        st.subheader("📚 Literature Evidence & Semantic Context Retrieval")
        st.write("Grounding citations extracted dynamically from the local **ChromaDB** vector database:")
        
        lit = state.get("literature") or {}
        sentences = lit.get("extracted_sentences", [])
        
        if sentences:
            for s in sentences:
                st.markdown(f'<div class="bio-card" style="border-left: 4px solid #3B82F6 !important;">💡 {s}</div>', unsafe_allow_html=True)
        else:
            st.info("No citations retrieved for this variant.")
            
        st.subheader("🔖 Key PubMed Publications Reference List")
        cons_ref = state.get("consensus", {}).get("references", [])
        if cons_ref:
            for ref in cons_ref:
                st.markdown(f"- 📄 {ref}")
        else:
            st.markdown("- None cached.")

else:
    # Blank State Landing Page
    st.info("👈 Select a target mutation from the sidebar and click 'Run Causal Curation' to generate reasoning traces.")
    
    # Platform Architecture Overview Graphic
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="bio-card" style="height:250px;">
            <h3>🤖 7-Agent Graph</h3>
            <p>LangGraph state machine orchestrating mutation analysis, sequence mapping, literature extraction, therapy curation, and validation reviews.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="bio-card" style="height:250px;">
            <h3>🕸️ NetworkX Knowledge Graph</h3>
            <p>Directed mapping linking entities: Mutation → Gene → Protein → Pathways → Disease. Traverses paths to discover therapeutic targets.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="bio-card" style="height:250px;">
            <h3>📚 ChromaDB RAG</h3>
            <p>Indexes programmatically queried PubMed abstracts on-the-fly and runs semantic similarity checks to ground all clinician recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
