# Multi-Source Candidate Data Transformer & Intelligence Platform

> **An Advanced, Research-Oriented Candidate Data Reconciliation, Active Learning, and Entity Intelligence System.**  
> Built for automated multi-source ingestion, explainable identity resolution, self-correcting AI extraction, knowledge graph analytics, and recruiter intelligence.

---

## 🌟 Key Capabilities & Research Features

### 1. Intelligent Candidate Reconciliation & Explainable Matching
- **Multi-Attribute Signal Extraction**: Computes matching vectors across 8 weighted signals (exact email, phone, LinkedIn, GitHub, fuzzy names with RapidFuzz token sort/set ratios, shared employers, skill Jaccard similarity, location).
- **Conflicting Hard Identifier Safeguard**: Prevents catastrophic false merges—if two candidates have distinct non-null emails or phone numbers, identity score drops to `0.0` regardless of name similarity.
- **Explainable Match Scoring**: Generates human-interpretable justifications and signal breakdowns for each proposed match.
- **Match Triaging**: Classifies pairs into **High-Confidence Matches** ($\ge 0.85$), **Uncertain Matches** ($0.50 - 0.85$), and **Likely Non-Matches** ($< 0.50$).

### 2. Active Learning & Feedback-Driven Matching
- **Review Queue**: Prioritizes pairs nearest the classification decision boundary (maximum information gain / uncertainty).
- **Temporal Split Validation**: Partitions reviewer feedback chronologically ($70\%$ Train, $15\%$ Validation, $15\%$ Held-out Test) to prevent future-to-past data leakage.
- **Threshold Recalibration**: Uses validation split grid-search to optimize F1 score and false-merge rates without risking catastrophic overfitting.
- **Configuration Versioning**: Stores versioned matching parameters with rollback capability.

### 3. Extractor–Validator–Refiner (EVR) Self-Correction Workflow
- **Multi-Format Document Parsing**: Ingests PDF (`pdfplumber`), DOCX (`python-docx`), and plain text resumes and recruiter notes.
- **Bounded Correction Loop**: Executes a deterministic 3-cycle EVR loop:
  1. *Extract*: LLM or rule-based parser extracts structured candidate attributes.
  2. *Validate*: Deterministically verifies dates (chronology), emails (RFC compliance), phone (E.164), and anchors facts against document text.
  3. *Refine*: Automatically repairs inverted dates, strips section header hallucinations, and flags unverifiable claims for human review.
- **Strict Anti-Hallucination**: Does not extrapolate or fabricate missing information.

### 4. Field-Level Provenance & Traceability
- **Evidence Spans**: Preserves character offsets (`start_char`, `end_char`), page references, and context snippets for extracted values.
- **Alternative Values**: Preserves unselected conflicting candidate values along with conflict resolution logs.
- **Decision Audit Trail**: Records exact resolution reasons (e.g. *Highest Confidence*, *Majority Vote*, *Latest Wins*, *Priority Order*, or *Manual Reviewer Override*).

### 5. Candidate Knowledge Graph (NetworkX)
- **Multi-Relational Graph**: Maps candidates to skills (`HAS_SKILL`), employers (`WORKED_AT`), titles (`HELD_TITLE`), and universities (`STUDIED_AT`).
- **Peer Insights**: Automatically infers cross-candidate connections (`SHARES_SKILLS`, `COMMON_EMPLOYER`).
- **Interactive Force-Directed Visualization**: Integrated into the full-stack web UI for neighborhood exploration.

### 6. Job Description Semantic Matching
- **Requirement Extraction**: Parses job descriptions into required skills, preferred skills, minimum experience years, and degrees.
- **Transparent Verification**: Categorizes requirements into *Verified*, *Partial*, and *Missing* with cited evidence.
- **Advisory-Only Policy**: Generates transparent fit scores without automated candidate disqualification.

### 7. Enterprise Security & Role-Based Access Control (RBAC)
- **JWT Authentication**: Direct bcrypt password hashing and OAuth2-compatible token generation.
- **Three Core Roles**:
  - `admin`: Full administrative control, user provisioning, matching configuration recalibration, and profile rollbacks.
  - `reviewer`: Review queue access, candidate merging/splitting, manual conflict resolution, and unmasked candidate inspection.
  - `recruiter`: Candidate search, job description matching, and role-based PII masking (redacted emails and phone numbers).
- **File Validation**: Strict MIME/magic-byte checks, extension whitelisting, and a 10MB per-file upload limit.

---

## 🏛 Architecture Overview

```
                                  +-----------------------------+
                                  |   Web UI (Tailwind + JS)    |
                                  +--------------+--------------+
                                                 | REST / JWT
                                                 v
+-----------------------------------------------------------------------------------------------+
|                                    FastAPI Application (/v1)                                  |
|  +--------------------+  +--------------------+  +---------------------+  +----------------+  |
|  |   Auth & RBAC      |  | Profile Explorer   |  |   Review Queue      |  | Quality Metrics|  |
|  | (/v1/auth)         |  | (/v1/candidates)   |  | (/v1/matching)      |  | (/v1/analytics)|  |
|  +--------------------+  +--------------------+  +---------------------+  +----------------+  |
|  +--------------------+  +--------------------+  +---------------------+  +----------------+  |
|  | Conflict Engine    |  | Knowledge Graph    |  |  JD Matcher         |  | EVR Extractor  |  |
|  | (/v1/conflict)     |  | (/v1/graph)        |  | (/v1/jobs)          |  | (/v1/extraction|  |
|  +--------------------+  +--------------------+  +---------------------+  +----------------+  |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                 +------------------------------+------------------------------+
                 v                                                             v
+----------------------------------+                         +----------------------------------+
|      Pipeline Processing Engine  |                         |    Persistence & Storage         |
| - Adapters (Resume, ATS, CSV, etc)|                         | - SQLite (Local Development)     |
| - Normalizers (Name, Date, etc.) |                         | - PostgreSQL (Production Engine) |
| - Deduplication & Graph Builder  |                         | - Alembic Migration Tooling      |
+----------------------------------+                         +----------------------------------+
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/DATATRANSFORMERCLONE.git
cd DATATRANSFORMERCLONE

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install package in editable mode with dependencies
pip install -e .
```

### 2. Configure Environment
```bash
# Copy example configuration
cp .env.example .env
```

### 3. Launch the Server
**Using convenience scripts:**
- On Windows: `.\start_api.bat` or `.\start_api.ps1`
- Direct CLI:
```bash
uvicorn data_transformer.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access Web Interface & Documentation
- **Interactive Web App**: [http://localhost:8000/ui/index.html](http://localhost:8000/ui/index.html)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 👥 Default Accounts (Local Development)

| Username | Password | Role | Permissions |
|---|---|---|---|
| `admin` | `admin123` | Administrator | System config, user management, rollback, unmasked PII |
| `reviewer` | `reviewer123` | Reviewer | Review queue, merge/split, manual conflict resolution |
| `recruiter` | `recruiter123` | Recruiter | Candidate search, JD matching, masked PII |

---

## 🧪 Comprehensive Test Suite

The system includes an extensive automated test suite covering unit, integration, and security scenarios across all pipeline components.

```bash
# Run all tests
pytest

# Run tests with summary report
pytest --tb=short -q

# Run specific modules
pytest tests/test_explainable_scorer.py
pytest tests/test_active_learning.py
pytest tests/test_evr_workflow.py
pytest tests/test_pii_masker.py
pytest tests/test_knowledge_graph.py
pytest tests/test_auth_api.py
pytest tests/test_db_repositories.py
```

---

## 📊 Evaluation & Research Benchmarks

The platform provides built-in metrics evaluation for entity resolution:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **False Merge Rate (FMR)**: $\frac{FP}{TP + FP}$

Threshold recalibration tracks before-and-after F1 and false-merge rates across active learning feedback cycles to provide measurable verification of matching performance.
