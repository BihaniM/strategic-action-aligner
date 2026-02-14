# Deployment Architecture (Detailed)

This document is the source-of-truth Markdown for rebuilding the deployment architecture diagram.

## Mermaid Diagram

```mermaid
flowchart LR
    %% =========================
    %% CLIENT
    %% =========================
    USER[End User Browser]

    %% =========================
    %% STREAMLIT CLOUD
    %% =========================
    subgraph SC[Streamlit Community Cloud]
        APP[Streamlit App Container\napp/streamlit_app.py]
        CODE[Application Code\napp/ + src/]
        ENV[Secrets + Env Vars\nHF_TOKEN, model IDs, CHROMA_COLLECTION]
        RUNTIME[Python Runtime\nruntime.txt + requirements.txt]
        LOCAL_CHROMA[(Local Chroma Persist\nwithin app filesystem)]
    end

    %% =========================
    %% EXTERNAL SERVICES
    %% =========================
    subgraph EXT[External Managed Services]
        HFAPI[Hugging Face Inference API\nhttps://api-inference.huggingface.co/models]
    end

    %% =========================
    %% SOURCE CONTROL / CI
    %% =========================
    subgraph REPO[GitHub Repository]
        MAIN[Main Branch]
        DOCS[docs/*.md architecture sources]
    end

    %% =========================
    %% CONNECTIONS
    %% =========================
    USER -->|HTTPS| APP
    APP --> CODE
    APP --> ENV
    APP --> RUNTIME

    APP -->|Read CSV uploads| APP
    APP -->|Vector read/write| LOCAL_CHROMA

    APP -->|HTTPS API calls\nembeddings + generation| HFAPI

    MAIN -->|Deploy from repo| APP
    DOCS --> MAIN

    %% =========================
    %% STYLES
    %% =========================
    classDef client fill:#E3F2FD,stroke:#1565C0,stroke-width:1px,color:#0D47A1;
    classDef cloud fill:#E8F5E9,stroke:#2E7D32,stroke-width:1px,color:#1B5E20;
    classDef ext fill:#FCE4EC,stroke:#AD1457,stroke-width:1px,color:#880E4F;
    classDef repo fill:#FFF3E0,stroke:#EF6C00,stroke-width:1px,color:#E65100;

    class USER client;
    class APP,CODE,ENV,RUNTIME,LOCAL_CHROMA cloud;
    class HFAPI ext;
    class MAIN,DOCS repo;
```

## Deployment Notes

- No local model host, tunnel, or ngrok is needed.
- Inference dependency is externalized to Hugging Face.
- Streamlit secrets are the only required runtime credentials/config layer.
- This deployment is optimized for a single, deterministic provider path.
