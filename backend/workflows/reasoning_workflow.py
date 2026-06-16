from langgraph.graph import StateGraph, END
from backend.schemas.agent_states import BioReasonState
from backend.agents.biomedical_agents import BiomedicalAgents
from typing import Dict, Any

class BioReasonWorkflow:
    """
    Defines the LangGraph orchestration state machine.
    Sequentially executes: Mutation -> Gene -> Pathway -> Literature -> Therapy -> Validation -> Consensus
    """
    
    def __init__(self, agents: BiomedicalAgents):
        self.agents = agents
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        # 1. Initialize StateGraph with the shared schema
        builder = StateGraph(BioReasonState)
        
        # 2. Add node functions mapping to our Agent calls
        builder.add_node("mutation_node", self.agents.run_mutation_analysis)
        builder.add_node("gene_node", self.agents.run_gene_protein_impact)
        builder.add_node("pathway_node", self.agents.run_pathway_analysis)
        builder.add_node("literature_node", self.agents.run_literature_intelligence)
        builder.add_node("therapy_node", self.agents.run_therapy_matching)
        builder.add_node("validation_node", self.agents.run_evidence_validation)
        builder.add_node("consensus_node", self.agents.run_consensus_aggregation)
        
        # 3. Define transitions (sequential edges)
        builder.set_entry_point("mutation_node")
        builder.add_edge("mutation_node", "gene_node")
        builder.add_edge("gene_node", "pathway_node")
        builder.add_edge("pathway_node", "literature_node")
        builder.add_edge("literature_node", "therapy_node")
        builder.add_edge("therapy_node", "validation_node")
        builder.add_edge("validation_node", "consensus_node")
        builder.add_edge("consensus_node", END)
        
        # 4. Compile the state machine
        return builder.compile()

    def execute(self, query: str) -> BioReasonState:
        """Runs the compiled agent graph end-to-end for a given query."""
        initial_state = BioReasonState(
            mutation_query=query,
            reasoning_trace=[]
        )
        # Execute in LangGraph
        result = self.workflow.invoke(initial_state.model_dump())
        return BioReasonState.model_validate(result)
