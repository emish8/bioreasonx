# 🧬 BioReason-X | Mutation-to-Therapy Intelligence Platform

BioReason-X is an explainable multi-agent biomedical reasoning platform designed to assist clinicians, oncology researchers, and life science teams in understanding the molecular impact of genomic mutations and identifying evidence-grounded therapeutic targets.

Instead of operating as a generic medical chatbot, BioReason-X traces the complete biological causal chain:
**Genomic Mutation → Gene → Protein Impact → Signaling Pathway Disruption → Disease Association → Targeted Therapy.**

---

## 🚀 Key Features

*   **7-Agent LangGraph Workflow**: Sequential, stateful orchestration where specialized agents analyze variants, evaluate protein folding, trace Reactome pathways, read PubMed references, match drug targets, validate claims, and assemble a clinical consensus report.
*   **100% Real-World Data Integration**: Connects to public, keyless biomedical databases dynamically in real time:
    *   **NCBI ClinVar**: Variant classifications and pathogenicity annotations.
    *   **MyGene.info**: Official gene summaries, UniProt IDs, and Reactome/KEGG pathway lists.
    *   **DGIdb (Drug-Gene Interaction Database)**: Match FDA-approved and investigational drug agents.
    *   **NCBI PubMed**: Search and scrape scientific publications.
*   **Local ChromaDB Vector RAG**: Downloads research abstracts and builds an in-memory vector store on-the-fly to query semantic context and ground agent justifications.
*   **Relational Knowledge Graph**: Dynamically constructs directed network topologies in **NetworkX** representing the causal biological path, rendered as an interactive, draggable Plotly graph.
*   **Vocal Summary Reports**: Integrated text-to-speech engine synthesizes reports for quick clinician review.
*   **AMD Instinct MI300X Optimization**: Formulated for high-throughput local open-source LLM inference (e.g. `Qwen-2.5-14B-Instruct` or `DeepSeek-R1-Distill-Qwen-14B`) served via **vLLM with ROCm**.

---

## 📂 Project Structure

```text
mutation-to-therapy/
├── backend/
│   ├── agents/
│   │   └── biomedical_agents.py       # 7-Agent logic implementations
│   ├── api/
│   │   └── main.py                    # FastAPI server & endpoints
│   ├── graph/
│   │   └── knowledge_graph.py         # NetworkX construction & Plotly renders
│   ├── rag/
│   │   └── vector_store.py            # ChromaDB + SentenceTransformers
│   ├── schemas/
│   │   └── agent_states.py            # Pydantic state declarations
│   └── services/
│       ├── data_fetcher.py            # ClinVar, MyGene, DGIdb, & PubMed APIs
│       ├── audio_processor.py         # gTTS Speech synthesis & transcribing
│       └── llm_client.py              # local vLLM / Cloud API routing
├── frontend/
│   ├── assets/
│   │   └── style.css                  # Custom biotech dark CSS
│   └── app.py                         # Streamlit multi-tab dashboard UI
├── data/
│   └── cached_mutations.json          # Cached data for 5 core oncology mutations
├── tests/
│   └── test_pipeline.py               # Automated pipeline validation suite
├── Dockerfile                         # App container packaging
├── docker-compose.yml                 # Service volume & port mapping
├── requirements.txt                   # Dependency declarations
├── start.sh                           # Concurrently launches API and Streamlit
└── .env                               # Port settings & model provider config
```

---

## ⚡ Setup & Quickstart

### Prerequisites
*   Python 3.10+
*   FastAPI dependencies and PortAudio (for audio utilities)
    *   **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev libasound2-dev`
    *   **Windows**: Automatically handled by python binaries

### 1. Installation
Clone the repository and install packages:
```bash
# Activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration
Copy the configuration environment variables from `.env` and choose your preferred `LLM_PROVIDER` (`vllm`, `openai`, or `gemini`). If no keys/endpoints are configured, the client automatically defaults to **Mock Mode**, allowing you to run the entire pipeline offline with structured local mock parsing.

---

## 🛠️ Running the Application

### Option A: Running Locally (Recommended for Development)
Start the FastAPI backend server first:
```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

In a second terminal, start the Streamlit client:
```bash
python -m streamlit run frontend/app.py
```
*   *Note: If the Streamlit application is launched without the FastAPI server running, it will automatically fall back to direct import execution and run the backend pipelines in-process, guaranteeing 100% uptime.*

### Option B: Running with Docker (For Staging & Production)
Build and run the multi-port container:
```bash
docker-compose up --build
```
Access the services at:
*   **FastAPI Swagger Docs**: `http://localhost:8000/docs`
*   **Streamlit UI**: `http://localhost:8501`

### Option C: Running in Jupyter Lab / Cloud Proxies
If you are running in a Jupyter Lab cloud pod (e.g. SageMaker, Saturn Cloud, Vertex AI):
1. **Launch the stack**:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
2. **Access the application**: Since you cannot access localhost directly, append the proxy suffix to your Jupyter base URL (make sure to include the trailing slash):
   *   **Streamlit UI (Port 8501)**: `https://<YOUR-JUPYTER-BASE-URL>/proxy/8501/`
   *   **FastAPI Docs (Port 8000)**: `https://<YOUR-JUPYTER-BASE-URL>/proxy/8000/docs`

---

## 🤖 AMD MI300X vLLM Setup (192 GB VRAM)
To serve models locally using ROCm, launch the vLLM servers in your GPU container. 

> [!IMPORTANT]
> The Reasoning/Text model is served on port **`8002`** to avoid conflicting with the FastAPI backend on port `8000`.

#### 1. Start the Text/Reasoning model (Port 8002)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --port 8002 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 8192
```

#### 2. Start the Vision model (Port 8001)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct-AWQ \
    --port 8001 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 8192
```

---

## 🧪 Automated Testing
Verify the entire agent workflow execution and Pydantic validation targets:
```bash
python tests/test_pipeline.py
```
