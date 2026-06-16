from backend.schemas.agent_states import (
    BioReasonState, MutationAnalysisState, GeneProteinState,
    PathwayState, LiteratureState, TherapyState, ValidationState, ConsensusState
)
from backend.services.llm_client import LLMClient
from backend.services.data_fetcher import DataFetcher
from backend.graph.knowledge_graph import BiomedicalKnowledgeGraph
from backend.rag.vector_store import BiomedicalVectorStore
from typing import Dict, Any, List

class BiomedicalAgents:
    """
    Implements the 7 collaborative reasoning agents.
    Each agent takes the global state, processes it using the LLM Client and APIs,
    updates the state, and appends a step to the reasoning trace.
    """
    
    def __init__(self, llm_client: LLMClient, data_fetcher: DataFetcher, 
                 graph: BiomedicalKnowledgeGraph, vector_store: BiomedicalVectorStore):
        self.llm = llm_client
        self.fetcher = data_fetcher
        self.graph = graph
        self.vector_store = vector_store

    def run_mutation_analysis(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 1: Mutation Analysis Agent
        Extracts details and reviews pathogenicity of the mutation.
        """
        # Fetch raw biomedical data
        raw_data = self.fetcher.get_mutation_data(state.mutation_query)
        
        prompt = f"""
        Analyze this genomic mutation using the fetched database records:
        Query: {state.mutation_query}
        ClinVar Pathogenicity: {raw_data.get('pathogenicity')}
        Mutation Type: {raw_data.get('mutation_type')}
        Chromosome Location: {raw_data.get('chromosome')}
        Clinical Description: {raw_data.get('clinical_significance')}
        
        Extract the gene symbol, official nomenclature representation, variant type, pathogenicity, and generate a brief clinical summary.
        """
        
        system_prompt = "You are an expert clinical genomic analyzer. Perform variant curation."
        analysis = self.llm.generate_json(prompt, system_prompt, MutationAnalysisState)
        
        # Build base knowledge graph nodes
        self.graph.build_from_mutation_data(state.mutation_query, raw_data)
        
        trace = f"Step 1: Mutation parsed as {analysis.mutation_type} in gene {analysis.gene_symbol}. Classification: {analysis.pathogenicity}."
        
        return {
            "raw_bio_data": raw_data,
            "mutation_analysis": analysis,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_gene_protein_impact(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 2: Gene/Protein Agent
        Details structural changes on UniProt proteins.
        """
        raw_data = state.raw_bio_data or {}
        analysis = state.mutation_analysis
        
        prompt = f"""
        Detail the molecular and functional protein impact of this mutation:
        Gene: {analysis.gene_symbol} ({analysis.clinical_summary})
        UniProt/RefSeq info: {raw_data.get('protein_impact', {}).get('consequence')}
        UniProt ID: {raw_data.get('protein_impact', {}).get('uniprot_id')}
        
        Identify the UniProt ID, affected structural domains, functional consequence (e.g. loss/gain-of-function), and molecular mechanism.
        """
        
        system_prompt = "You are a biochemical sequence and structural biology agent."
        protein_state = self.llm.generate_json(prompt, system_prompt, GeneProteinState)
        
        trace = f"Step 2: UniProt ID {protein_state.uniprot_id} mapped. Molecular consequence: {protein_state.functional_consequence}."
        
        return {
            "gene_protein": protein_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_pathway_analysis(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 3: Pathway Agent
        Analyzes disrupted Reactome signaling cascades.
        """
        raw_data = state.raw_bio_data or {}
        gene_symbol = state.mutation_analysis.gene_symbol
        
        pathways_text = "\n".join([f"- {p.get('name')} (ID: {p.get('id')})" for p in raw_data.get('pathways', [])])
        
        prompt = f"""
        Analyze the signaling pathway disruption caused by mutation in gene {gene_symbol}.
        Linked Reactome Pathways:
        {pathways_text}
        
        Determine the affected pathways, cellular phenotype impact, and downstream effector targets (e.g. transcription factor triggers).
        """
        
        system_prompt = "You are a systems biology and cellular pathway analysis agent."
        pathway_state = self.llm.generate_json(prompt, system_prompt, PathwayState)
        
        trace = f"Step 3: Pathway analysis completed. Cellular impact: {pathway_state.cellular_impact}."
        
        return {
            "pathway": pathway_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_literature_intelligence(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 4: Literature Agent
        Indexes PubMed papers and extracts sentences.
        """
        raw_data = state.raw_bio_data or {}
        pubmed_articles = raw_data.get("pubmed", [])
        
        # Index retrieved articles into ChromaDB for dynamic RAG grounding
        self.vector_store.add_publications(pubmed_articles, state.mutation_query)
        
        # Search relevant sentences from vector store
        query = f"{state.mutation_analysis.gene_symbol} {state.mutation_analysis.variant_nomenclature} therapeutic response survival"
        hits = self.vector_store.search_evidence(query, limit=3)
        
        pmids = [hit["metadata"]["pmid"] for hit in hits]
        sentences = [f"\"{hit['text']}\" (PMID: {hit['metadata']['pmid']})" for hit in hits]
        
        prompt = f"""
        Review the following evidence extractions for mutation {state.mutation_query}:
        Extractions:
        {chr(10).join(sentences)}
        
        Verify the strength of the clinical evidence supporting therapeutic efficacy or resistance markers.
        """
        
        system_prompt = "You are a scientific literature intelligence agent."
        lit_state = self.llm.generate_json(prompt, system_prompt, LiteratureState)
        lit_state.relevant_pmids = list(set(pmids))
        lit_state.extracted_sentences = sentences
        
        trace = f"Step 4: Indexed {len(pubmed_articles)} publications in ChromaDB. Retrieved {len(sentences)} clinical evidence sentences."
        
        return {
            "literature": lit_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_therapy_matching(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 5: Therapy Agent
        Identifies drugs using target datasets and literature.
        """
        raw_data = state.raw_bio_data or {}
        gene_symbol = state.mutation_analysis.gene_symbol
        drugs_data = raw_data.get("drugs", [])
        
        drugs_text = "\n".join([f"- {d.get('name')} ({d.get('type')}): {d.get('indication')}" for d in drugs_data])
        
        prompt = f"""
        Identify targeted therapies for a mutation in gene {gene_symbol}.
        DGIdb Drug Target Interactions:
        {drugs_text}
        
        Literature evidence extractions:
        {state.literature.extracted_sentences}
        
        Match specific drug names, explain their molecular mechanism of action, and list any secondary resistance risks.
        """
        
        system_prompt = "You are a clinical pharmacogenomics agent."
        therapy_state = self.llm.generate_json(prompt, system_prompt, TherapyState)
        
        trace = f"Step 5: Identified {len(therapy_state.recommended_therapies)} matched therapies: {', '.join(therapy_state.recommended_therapies)}."
        
        return {
            "therapy": therapy_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_evidence_validation(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 6: Evidence Validation Agent
        Cross-references claims and scores evidence.
        """
        # Retrieve facts to check
        claims = f"Therapies: {state.therapy.recommended_therapies}. Rationale: {state.therapy.mechanism_of_action}."
        
        # Query ChromaDB specifically to validate this therapeutic claim
        validation_hits = self.vector_store.search_evidence(claims, limit=3)
        valid_text = "\n".join([f"- {h['text']} (Similarity: {h['similarity']})" for h in validation_hits])
        
        prompt = f"""
        Perform a validation audit on these claims against the retrieved literature context.
        Claims: {claims}
        Retrieved Context:
        {valid_text}
        
        Check for errors or contradictions. Score the validation confidence (0-100), identify contradictions, and rate the evidence base.
        """
        
        system_prompt = "You are a medical evidence review board agent."
        validation_state = self.llm.generate_json(prompt, system_prompt, ValidationState)
        
        trace = f"Step 6: Evidence validated with confidence score: {validation_state.validation_score}%. Rating: {validation_state.evidence_rating}."
        
        return {
            "validation": validation_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }

    def run_consensus_aggregation(self, state: BioReasonState) -> Dict[str, Any]:
        """
        Agent 7: Consensus Agent
        Aggregates outputs into an explainable report.
        """
        prompt = f"""
        Synthesize the final report for the mutation {state.mutation_query}:
        - Analysis: {state.mutation_analysis.clinical_summary} ({state.mutation_analysis.pathogenicity})
        - Protein Impact: {state.gene_protein.functional_consequence} ({state.gene_protein.molecular_mechanism})
        - Cellular Pathways: {state.pathway.affected_pathways}
        - Matched Drugs: {state.therapy.recommended_therapies} (Mechanism: {state.therapy.mechanism_of_action})
        - Validation Score: {state.validation.validation_score}% ({state.validation.evidence_rating})
        
        Create a final consensus statement, state the final confidence rating percentage, list actionable clinical steps, and compile references.
        """
        
        system_prompt = "You are the clinical board director agent compiling the consensus report."
        consensus_state = self.llm.generate_json(prompt, system_prompt, ConsensusState)
        
        trace = f"Step 7: Consensus achieved. Final report compiled. Confidence: {consensus_state.confidence_rating}."
        
        return {
            "consensus": consensus_state,
            "reasoning_trace": state.reasoning_trace + [trace]
        }
