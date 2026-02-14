# System Architecture (Detailed)

This document is the source-of-truth Markdown for rebuilding the system architecture diagram.

## Mermaid Diagram

```mermaid
flowchart TB
    %% =========================
    %% LAYERS
    %% =========================
    subgraph U[User Layer]
        USER[Business User]
    end

    subgraph P[Presentation Layer]
        UI[Streamlit Dashboard\napp/streamlit_app.py]
    end

    subgraph A[Application Layer]
        ORCH[Pipeline Orchestrator\nsrc/full_pipeline.py]

        DL[Data Loader + Chunk Builder\nsrc/data_loader.py]
        EMB[Embedding Engine\nsrc/embedding_engine.py]
        VS[Vector Store Adapter\nsrc/vector_store.py]
        AE[Alignment Engine\nsrc/alignment_engine.py]

        IA[Improvement Agent\nsrc/improvement_agent.py]
        AR[Agentic Reasoning Layer\nsrc/agentic_reasoning.py]
        EV[Evaluation Engine\nsrc/evaluation.py]

        HFC[Hugging Face Client\nsrc/hf_client.py]
        CFG[Configuration\nsrc/config.py]
    end

    subgraph D[Data Layer]
        IN[(Input CSVs\nstrategic_plan.csv\naction_plan.csv)]
        GT[(Optional Ground Truth\nground_truth.csv)]
        CH[(ChromaDB\nchunk embeddings + metadata)]
        OUT[(Output Artifacts\nCSV + JSONL + summary)]
    end

    subgraph S[External Service]
        HF[Hugging Face Inference API\nEmbeddings + Chat/Generation]
    end

    %% =========================
    %% PRIMARY FLOW
    %% =========================
    USER --> UI
    UI --> ORCH

    ORCH --> DL
    DL --> EMB
    EMB --> AE
    AE --> IA
    IA --> AR
    ORCH --> EV

    %% =========================
    %% DATA INTERACTIONS
    %% =========================
    IN --> DL
    GT --> EV

    EMB --> VS
    AE --> VS
    VS --> CH
    CH --> VS

    ORCH --> OUT
    EV --> OUT

    %% =========================
    %% SHARED SERVICES
    %% =========================
    CFG --> ORCH
    CFG --> EMB
    CFG --> IA
    CFG --> AR
    CFG --> HFC

    EMB --> HFC
    IA --> HFC
    AR --> HFC
    HFC --> HF

    %% =========================
    %% STYLES
    %% =========================
    classDef user fill:#E3F2FD,stroke:#1565C0,stroke-width:1px,color:#0D47A1;
    classDef present fill:#F3E5F5,stroke:#6A1B9A,stroke-width:1px,color:#4A148C;
    classDef app fill:#E8F5E9,stroke:#2E7D32,stroke-width:1px,color:#1B5E20;
    classDef data fill:#FFF3E0,stroke:#EF6C00,stroke-width:1px,color:#E65100;
    classDef ext fill:#FCE4EC,stroke:#AD1457,stroke-width:1px,color:#880E4F;

    class USER user;
    class UI present;
    class ORCH,DL,EMB,VS,AE,IA,AR,EV,HFC,CFG app;
    class IN,GT,CH,OUT data;
    class HF ext;
```

## Diagram Notes

- Single-provider architecture: Hugging Face only.
- `src/hf_client.py` is the single model gateway for embedding and generation calls.
- ChromaDB stores chunk-level vectors and metadata for retrieval/matching.
- Agentic and improvement stages operate only on low-alignment pairs.
- Evaluation runs when optional ground-truth mapping is provided.
