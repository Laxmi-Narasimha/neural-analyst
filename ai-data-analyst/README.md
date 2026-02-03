# AI Enterprise Data Analyst

**Version 3.0.0** | **242+ Features** | **12+ MNC Techniques**

> The most comprehensive AI-powered Enterprise Data Analyst system, incorporating cutting-edge techniques from 12+ of the world's leading technology companies.

## 🚀 Features

### Core Capabilities
- **Data Ingestion**: Multi-format support (CSV, Excel, JSON, Parquet) with auto-detection
- **Data Quality**: Profiling, lineage tracking, anomaly detection
- **Exploratory Data Analysis**: Automated EDA with statistical insights
- **Machine Learning**: AutoML, classification, regression, clustering
- **Deep Learning**: Neural networks, LSTMs, Transformers
- **Time Series**: Forecasting with Prophet, N-BEATS, TFT
- **Natural Language Processing**: Sentiment analysis, NER, text classification
- **LLM Integration**: Conversational AI with GPT-4o

### MNC Techniques Integrated
| Company | Techniques |
|---------|------------|
| **Netflix** | Contextual bandits, A/B experimentation |
| **Uber** | H3 geospatial, surge pricing, DeepETA |
| **Airbnb** | Aerosolve dynamic pricing, Journey Ranker |
| **LinkedIn** | LiGNN Graph Neural Networks |
| **Spotify** | Audio feature extraction, collaborative filtering |
| **Stripe/PayPal** | Radar ML fraud detection |

## 📦 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Node.js 18+ (for frontend)

### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration (especially OPENAI_API_KEY)

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

### Environment Variables
Key environment variables to configure:
```
OPENAI_API_KEY=sk-your-key-here
DB_HOST=localhost
DB_NAME=ai_data_analyst
DB_PASSWORD=your-password
SECRET_KEY=your-secret-key-min-32-chars
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│           React Dashboard  |  Chat Interface                │
├─────────────────────────────────────────────────────────────┤
│                    API LAYER (FastAPI)                       │
│    Datasets  |  Analyses  |  Chat  |  ML Models  |  Auth   │
├─────────────────────────────────────────────────────────────┤
│                    AGENT ORCHESTRATION                       │
│  Orchestrator  |  EDA  |  Stats  |  ML  |  NLP  |  Viz     │
├─────────────────────────────────────────────────────────────┤
│                    SERVICE LAYER                             │
│  LLM Service  |  Data Ingestion  |  ML Engine  |  Cache    │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                │
│   PostgreSQL  |  Pinecone (Vectors)  |  Redis (Cache)       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai-data-analyst/
├── backend/
│   ├── app/
│   │   ├── core/           # Config, exceptions, logging
│   │   ├── models/         # SQLAlchemy models
│   │   ├── services/       # Business logic
│   │   ├── agents/         # AI agents (ReAct pattern)
│   │   ├── api/            # FastAPI routes
│   │   ├── ml/             # ML/DL modules
│   │   └── main.py         # Application entry
│   ├── tests/              # Test suite
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   └── src/                # React application
└── docs/                   # Documentation
```

## 🔌 API Endpoints

### Datasets
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets` - List datasets
- `GET /api/v1/datasets/{id}` - Get dataset details
- `POST /api/v1/datasets/{id}/process` - Process dataset

### Chat
- `POST /api/v1/chat` - Send message to AI
- `GET /api/v1/chat/conversations` - List conversations
- `GET /api/v1/chat/conversations/{id}` - Get conversation

### Health
- `GET /health` - Health check
- `GET /ready` - Readiness check

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific tests
pytest tests/test_datasets.py -v
```

## 📊 Key Design Patterns

- **Repository Pattern**: Data access abstraction
- **Factory Pattern**: Object creation
- **Singleton Pattern**: Configuration management
- **Strategy Pattern**: Interchangeable algorithms
- **Template Method**: Base agent execution flow
- **ReAct Pattern**: Agent reasoning and acting

## 🔒 Security Features

- JWT authentication with refresh tokens
- Rate limiting (configurable)
- CORS protection
- Input validation (Pydantic)
- SQL injection prevention (SQLAlchemy ORM)
- PII detection and warnings

## 📈 Production Considerations

- Connection pooling (PostgreSQL)
- Async operations throughout
- Structured JSON logging
- Health/readiness endpoints
- Docker support
- Celery for background tasks

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.
