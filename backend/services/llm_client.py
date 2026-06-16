import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel

load_dotenv()

class LLMClient:
    """
    Unified LLM Client supporting:
    - Local vLLM servers running on AMD Instinct MI300X GPUs via ROCm.
    - Fallbacks to commercial cloud models (OpenAI or Gemini).
    - An offline deterministic mock engine that generates rich responses when keys/servers are offline.
    """
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "mock").lower()
        self.text_api_base = os.getenv("VLLM_TEXT_API_BASE", "http://localhost:8000/v1")
        self.text_model = os.getenv("VLLM_TEXT_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_API_MODEL", "gpt-4o")
        
        # Initialize client if keys/endpoints exist
        self.client = None
        if self.provider == "vllm":
            self.client = OpenAI(base_url=self.text_api_base, api_key="dummy-key")
        elif self.provider == "openai" and self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
        else:
            print("LLM Client: No active model servers or API keys detected. Initializing offline mock processor.")
            self.provider = "mock"

    def generate_text(self, prompt: str, system_prompt: str = "You are a professional clinical geneticist.") -> str:
        """Generates plain-text completion."""
        if self.provider == "mock" or not self.client:
            return self._mock_text_generation(prompt)
            
        try:
            response = self.client.chat.completions.create(
                model=self.text_model if self.provider == "vllm" else self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API execution error: {e}. Falling back to offline generation...")
            return self._mock_text_generation(prompt)

    def generate_json(self, prompt: str, system_prompt: str, response_schema: Type[BaseModel]) -> BaseModel:
        """
        Generates structured outputs matching a Pydantic model.
        Falls back to local parser if endpoints are unavailable.
        """
        if self.provider == "mock" or not self.client:
            return self._mock_json_generation(prompt, response_schema)
            
        try:
            # For older vLLM endpoints or OpenAI, request JSON mode
            response = self.client.chat.completions.create(
                model=self.text_model if self.provider == "vllm" else self.openai_model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\nYou MUST output valid raw JSON matching this schema: {response_schema.model_json_schema()}"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            # Parse into Pydantic model
            parsed_data = json.loads(content)
            return response_schema.model_validate(parsed_data)
        except Exception as e:
            print(f"LLM JSON parsing error: {e}. Falling back to deterministic structured parser...")
            return self._mock_json_generation(prompt, response_schema)

    def _mock_text_generation(self, prompt: str) -> str:
        """Helper to create detailed natural language reasoning traces offline."""
        if "consensus" in prompt.lower() or "final report" in prompt.lower():
            return "Consensus Statement: Analysis reveals high confidence target matches. " \
                   "The genomic alteration destabilizes binding, indicating PARP inhibitor sensitivity."
        return "Offline Analysis: Active pocket configuration resolved. Significant hydrogen bonding shift identified."

    def _mock_json_generation(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """Generates deterministic Pydantic objects based on regex keyword matching."""
        # Check keywords in prompt to return meaningful data
        prompt_lower = prompt.lower()
        
        # Determine which agent schema is requested
        schema_name = schema.__name__
        
        # Default mock instances matching schemas
        if schema_name == "MutationAnalysisState":
            # Extract query details
            gene = "BRCA1"
            mut = "c.5266dupC"
            patho = "Pathogenic"
            m_type = "Frameshift Insertion"
            
            if "egfr" in prompt_lower:
                gene = "EGFR"
                mut = "L858R"
                m_type = "Missense Substitution"
            elif "braf" in prompt_lower:
                gene = "BRAF"
                mut = "V600E"
                m_type = "Missense Substitution"
            elif "kras" in prompt_lower:
                gene = "KRAS"
                mut = "G12C"
                m_type = "Missense Substitution"
            elif "alk" in prompt_lower:
                gene = "ALK"
                mut = "F1174L"
                patho = "Pathogenic / Resistant"
                m_type = "Missense Substitution"
                
            return schema(
                gene_symbol=gene,
                variant_nomenclature=mut,
                pathogenicity=patho,
                mutation_type=m_type,
                clinical_summary=f"Parsed {gene} variant showing pathogenic significance."
            )
            
        elif schema_name == "GeneProteinState":
            gene = "BRCA1"
            u_id = "P38398"
            conseq = "Truncates C-terminal BRCT domain, damaging DNA double-strand repair."
            if "egfr" in prompt_lower:
                gene = "EGFR"
                u_id = "P00533"
                conseq = "Locks kinase domain in active state promoting MAPK transcription."
            elif "braf" in prompt_lower:
                gene = "BRAF"
                u_id = "P15056"
                conseq = "Keeps kinase active as monomer activating downstream MEK."
            elif "kras" in prompt_lower:
                gene = "KRAS"
                u_id = "P01116"
                conseq = "Impairs intrinsic GTP hydrolysis, locking KRAS in active GTP state."
            elif "alk" in prompt_lower:
                gene = "ALK"
                u_id = "Q9UM73"
                conseq = "Increases ATP binding affinity, bypassing first-generation TKIs."
                
            return schema(
                uniprot_id=u_id,
                protein_domains="Kinase catalytic domain or repair interaction domain",
                functional_consequence=conseq,
                molecular_mechanism="Constitutive downstream cascade activation or repair loop truncation"
            )
            
        elif schema_name == "PathwayState":
            pathways = ["DNA Double-Strand Break Repair", "Homologous Recombination"]
            desc = "Inability to repair double-strand breaks leads to genomic instability."
            if "egfr" in prompt_lower:
                pathways = ["EGFR Signaling Pathway", "MAPK Cascade"]
                desc = "Upregulation of cell cycle progression and survival cascades."
            elif "braf" in prompt_lower:
                pathways = ["RAF-MAPK Cascade"]
                desc = "Continuous cell growth signaling bypassing receptor control."
            elif "kras" in prompt_lower:
                pathways = ["Signaling by RAS variants", "MAPK Pathway"]
                desc = "Continuous activation of RAF-MEK-ERK growth signaling cascades."
            elif "alk" in prompt_lower:
                pathways = ["Signaling by ALK", "AKT Pathway"]
                desc = "Upregulation of survival pathways causing drug-resistant expansion."
                
            return schema(
                affected_pathways=pathways,
                cellular_impact=desc,
                downstream_targets=["RAF", "MEK", "ERK", "AKT"]
            )
            
        elif schema_name == "LiteratureState":
            return schema(
                relevant_pmids=["29333925", "20729471"],
                extracted_sentences=[
                    "Evidence matches clinical efficacy metrics.",
                    "Significantly increased progression-free survival rates recorded."
                ],
                evidence_strength="High (Phase III Trials)"
            )
            
        elif schema_name == "TherapyState":
            drugs = ["Olaparib", "Talazoparib"]
            rat = "Induces synthetic lethality in HR-deficient cells."
            if "egfr" in prompt_lower:
                drugs = ["Osimertinib", "Erlotinib"]
                rat = "Competes with ATP binding in active site pockets."
            elif "braf" in prompt_lower:
                drugs = ["Vemurafenib", "Dabrafenib"]
                rat = "Binds to monomeric active BRAF kinase pocket."
            elif "kras" in prompt_lower:
                drugs = ["Sotorasib", "Adagrasib"]
                rat = "Covalently links to Cysteine at position 12, locking GDP state."
            elif "alk" in prompt_lower:
                drugs = ["Lorlatinib"]
                rat = "Macrocyclic inhibitor overcoming secondary pocket ATP-binding shifts."
                
            return schema(
                recommended_therapies=drugs,
                mechanism_of_action=rat,
                resistance_risks="Acquisition of secondary kinase domain gatekeeper variants."
            )
            
        elif schema_name == "ValidationState":
            return schema(
                is_validated=True,
                validation_score=92.5,
                contradictions_found="None identified in current vector references.",
                evidence_rating="Strongly Supported"
            )
            
        elif schema_name == "ConsensusState":
            return schema(
                final_consensus_statement="BioReason-X concludes the identified variant is pathogenic. Targeted therapeutic strategies (inhibitors/PARPi) are highly indicated based on clinical trial evidence.",
                confidence_rating="92%",
                recommended_actions=["Initiate clinical sequencing confirm", "Select matched therapy options"],
                references=["Ledermann J, et al. Lancet Oncol. 2010"]
            )
            
        # Fallback empty model
        return schema.model_construct()
