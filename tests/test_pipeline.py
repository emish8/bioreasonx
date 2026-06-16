import os
import sys

# Force pure-Python implementation for protobuf to resolve version mismatches in the environment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add root folder to sys.path to allow running this script from the project directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.llm_client import LLMClient
from backend.services.data_fetcher import DataFetcher
from backend.graph.knowledge_graph import BiomedicalKnowledgeGraph
from backend.rag.vector_store import BiomedicalVectorStore
from backend.agents.biomedical_agents import BiomedicalAgents
from backend.workflows.reasoning_workflow import BioReasonWorkflow

def test_causal_reasoning_flow():
    print("==================================================")
    print("BioReason-X: Initializing Automated pipeline checks...")
    print("==================================================")
    
    # 1. Initialize core system components
    llm = LLMClient()
    fetcher = DataFetcher(cache_file_path="data/cached_mutations.json")
    graph = BiomedicalKnowledgeGraph()
    vector_store = BiomedicalVectorStore(persist_dir="data/chroma_db_test")
    
    agents = BiomedicalAgents(llm, fetcher, graph, vector_store)
    workflow = BioReasonWorkflow(agents)
    
    print("[SUCCESS] Core services initialized.")
    
    # 2. Run reasoning workflow for testing mutation
    test_query = "BRCA1 c.5266dupC"
    print(f"\nExecuting 7-Agent LangGraph workflow for: '{test_query}'...")
    
    state = workflow.execute(test_query)
    
    # 3. Verify execution results
    print("\nVerifying state outputs:")
    
    assert state.mutation_analysis is not None, "Error: Mutation Analysis stage failed."
    print(f"[SUCCESS] Mutation Agent: Gene={state.mutation_analysis.gene_symbol}, Type={state.mutation_analysis.mutation_type}")
    
    assert state.gene_protein is not None, "Error: Protein Sequence stage failed."
    print(f"[SUCCESS] Protein Agent: UniProt ID={state.gene_protein.uniprot_id}")
    
    assert state.pathway is not None, "Error: Cellular Pathway stage failed."
    print(f"[SUCCESS] Pathway Agent: Disrupted pathways count={len(state.pathway.affected_pathways)}")
    
    assert state.therapy is not None, "Error: Therapy Mapping stage failed."
    print(f"[SUCCESS] Therapy Agent: Matched drugs={state.therapy.recommended_therapies}")
    
    assert state.validation is not None, "Error: Validation Curation stage failed."
    print(f"[SUCCESS] Validation Agent: Score={state.validation.validation_score}%, Rating={state.validation.evidence_rating}")
    
    assert state.consensus is not None, "Error: Consensus Curation stage failed."
    print(f"[SUCCESS] Consensus Agent: Status=Report Compiled.")
    
    # 4. Print trace
    print("\nChronological Agent Reasoning Trace Logs:")
    for step in state.reasoning_trace:
        # Strip potential emoji markers inside trace strings if any
        clean_step = step.replace("✔️", "[SUCCESS]").replace("🧬", "").replace("⚙️", "").replace("🤖", "")
        print(f"  {clean_step}")
        
    print("\nKnowledge Graph Verification:")
    print(f"  Graph Nodes Count: {len(graph.graph.nodes())}")
    print(f"  Graph Edges Count: {len(graph.graph.edges())}")
    assert len(graph.graph.nodes()) > 0, "Error: Graph nodes not generated."
    print("[SUCCESS] Knowledge Graph built successfully.")
    
    print("\n==================================================")
    print("All Automated Verification Checks Passed!")
    print("==================================================")

if __name__ == "__main__":
    test_causal_reasoning_flow()
