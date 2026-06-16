from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Agent Output Schemas ---

class MutationAnalysisState(BaseModel):
    gene_symbol: str = Field(description="The HGNC gene symbol affected (e.g. BRCA1)")
    variant_nomenclature: str = Field(description="Genomic or protein nomenclature of variation (e.g. c.5266dupC)")
    pathogenicity: str = Field(description="Pathogenicity classification (e.g. Pathogenic, Likely Pathogenic)")
    mutation_type: str = Field(description="Molecular type (e.g. Missense, Frameshift)")
    clinical_summary: str = Field(description="Short clinical overview of the variant")

class GeneProteinState(BaseModel):
    uniprot_id: str = Field(description="UniProt database identifier")
    protein_domains: str = Field(description="Affected protein domains and functional regions")
    functional_consequence: str = Field(description="Consequence of mutation on the protein's activity")
    molecular_mechanism: str = Field(description="Underlying molecular mechanism of impact")

class PathwayState(BaseModel):
    affected_pathways: List[str] = Field(default=[], description="Biological pathways disrupted or activated")
    cellular_impact: str = Field(description="Downstream cellular phenotype impact")
    downstream_targets: List[str] = Field(default=[], description="Downstream signaling target nodes")

class LiteratureState(BaseModel):
    relevant_pmids: List[str] = Field(default=[], description="PMIDs of retrieved supporting articles")
    extracted_sentences: List[str] = Field(default=[], description="Relevant text extractions from abstracts")
    evidence_strength: str = Field(description="Strength annotation (e.g. Phase III, In Vitro)")

class TherapyState(BaseModel):
    recommended_therapies: List[str] = Field(default=[], description="Matched targeted therapies")
    mechanism_of_action: str = Field(description="Molecular rationale for the drug response")
    resistance_risks: str = Field(description="Potential mechanism of secondary drug resistance")

class ValidationState(BaseModel):
    is_validated: bool = Field(description="True if validation checks pass")
    validation_score: float = Field(description="Confidence/validation score between 0 and 100")
    contradictions_found: str = Field(description="Description of any contradicting literature findings")
    evidence_rating: str = Field(description="Overall rating of evidence base (e.g., Strongly Supported)")

class ConsensusState(BaseModel):
    final_consensus_statement: str = Field(description="Explainable consensus summary")
    confidence_rating: str = Field(description="Formatted confidence score percentage")
    recommended_actions: List[str] = Field(default=[], description="Clinician next step actions")
    references: List[str] = Field(default=[], description="Formatted key references list")


# --- Global Graph State ---

class BioReasonState(BaseModel):
    mutation_query: str = Field(description="User genomic mutation input query")
    raw_bio_data: Optional[Dict[str, Any]] = Field(default=None, description="Raw fetched biomedical records")
    
    mutation_analysis: Optional[MutationAnalysisState] = Field(default=None)
    gene_protein: Optional[GeneProteinState] = Field(default=None)
    pathway: Optional[PathwayState] = Field(default=None)
    literature: Optional[LiteratureState] = Field(default=None)
    therapy: Optional[TherapyState] = Field(default=None)
    validation: Optional[ValidationState] = Field(default=None)
    consensus: Optional[ConsensusState] = Field(default=None)
    
    reasoning_trace: List[str] = Field(default=[], description="Chronological reasoning step trace list")
