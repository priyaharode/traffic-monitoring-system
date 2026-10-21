# Intelligent Traffic Monitoring System

An automated backend and monitoring system for traffic violation detection, evidence storage, and cryptographic integrity verification.

## Folder Structure

```text
traffic-monitoring-system/
├── backend/
│   ├── main.py              # FastAPI application endpoints and routers
│   ├── database.py          # SQLAlchemy database connection and session maker
│   ├── models.py            # Database models (Violation)
│   ├── schemas.py           # Pydantic request/response models and validation
│   ├── migrate_db.py        # Database migration utilities
│   ├── requirements.txt     # Backend dependencies
│   └── evidence/            # Local storage for captured violation evidence images
├── dataset/                 # Training and test datasets for computer vision models
├── frontend/                # Frontend monitoring dashboard
└── README.md                # Project documentation
```

## Features

- **FastAPI REST API**: High-performance asynchronous API for traffic violation logging and review.
- **Strict Query & Model Validation**: Restricted types for violation categories (`no_helmet`, `triple_riding`, `red_light`, `stop_line`) and lifecycle statuses (`pending`, `approved`, `rejected`).
- **Evidence Image Storage**: Handles multipart image uploads (JPEG/PNG up to 5 MB), validated for image authenticity via Pillow.
- **Cryptographic Evidence Verification**: Computes and stores SHA-256 fingerprints on upload, with an on-demand integrity verification endpoint to detect evidence tampering.
- **Safe Path Resolution**: Prevents path traversal vulnerabilities when streaming evidence files.
- **Robust Database Rollbacks**: Atomic transactions with automatic cleanup of newly created files upon database failure.

## Getting Started

### 1. Setup Virtual Environment

```bash
cd backend
py -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Windows (Command Prompt)**: `venv\Scripts\activate`
- **Linux/macOS**: `source venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Development Server

```bash
uvicorn main:app --reload
```

The API will be available at:
- Server: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API Docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
