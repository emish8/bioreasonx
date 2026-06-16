import os
# Force pure-Python implementation for protobuf to resolve version mismatches in the environment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List
from backend.services.llm_client import LLMClient
from backend.services.data_fetcher import DataFetcher
from backend.graph.knowledge_graph import BiomedicalKnowledgeGraph
from backend.rag.vector_store import BiomedicalVectorStore
from backend.agents.biomedical_agents import BiomedicalAgents
from backend.workflows.reasoning_workflow import BioReasonWorkflow
from backend.schemas.agent_states import BioReasonState

app = FastAPI(
    title="BioReason-X API",
    description="Explainable Biomedical Multi-Agent Reasoning Server",
    version="1.0.0"
)

# --- Global Orchestration Singleton Container ---
llm_client = LLMClient()
data_fetcher = DataFetcher()
knowledge_graph = BiomedicalKnowledgeGraph()
vector_store = BiomedicalVectorStore()
agents = BiomedicalAgents(llm_client, data_fetcher, knowledge_graph, vector_store)
workflow = BioReasonWorkflow(agents)


# --- API Input/Output Schemas ---

class QueryModel(BaseModel):
    query: str

class TherapyQuery(BaseModel):
    gene: str
    mutation: str


# --- FastAPI Endpoint Implementations ---

@app.get("/health")
def health_check():
    """Endpoint for monitoring API and service connection status."""
    return {
        "status": "healthy",
        "llm_provider": llm_client.provider,
        "chroma_collection": vector_store.collection.name,
        "active_graph_nodes": len(knowledge_graph.graph)
    }

@app.post("/analyze")
def analyze_mutation(payload: QueryModel):
    """
    Executes Phase 1: Extracts pathogenicity and nomenclature details.
    """
    try:
        # Create minimal state
        state = BioReasonState(mutation_query=payload.query)
        result = agents.run_mutation_analysis(state)
        return {
            "success": True,
            "analysis": result["mutation_analysis"].model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason")
def reason_pipeline(payload: QueryModel):
    """
    Executes the entire 7-agent LangGraph workflow from genomic mutation to therapeutic validation.
    """
    try:
        result = workflow.execute(payload.query)
        return {
            "success": True,
            "state": result.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/therapy")
def query_therapy(payload: QueryModel):
    """
    Direct endpoint to extract drug recommendation profiles.
    """
    try:
        # Run workflow up to therapy
        result = workflow.execute(payload.query)
        if result.therapy:
            return {
                "success": True,
                "therapies": result.therapy.recommended_therapies,
                "rationale": result.therapy.mechanism_of_action,
                "resistance_risks": result.therapy.resistance_risks
            }
        raise HTTPException(status_code=400, detail="Therapy recommendation stage was not completed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph_data():
    """
    Returns the serialized knowledge graph node-link dataset.
    """
    try:
        nodes = []
        for n in knowledge_graph.graph.nodes():
            attrs = knowledge_graph.graph.nodes[n]
            nodes.append({
                "id": n,
                "type": attrs.get("type", "Unknown"),
                "label": attrs.get("label", n),
                "color": attrs.get("color", "#64748B")
            })
            
        links = []
        for u, v in knowledge_graph.graph.edges():
            rel = knowledge_graph.graph.edges[u, v].get("relation", "links")
            links.append({
                "source": u,
                "target": v,
                "relation": rel
            })
            
        return {
            "success": True,
            "nodes": nodes,
            "links": links
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
