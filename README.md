# RAG Benchmark Platform

A comprehensive web platform for evaluating Retrieval-Augmented Generation (RAG) pipelines across multiple languages, document types, and evaluation metrics.

## Features

- **Multi-Lingual Support**: Test pipelines on English, Hindi, and Telugu
- **Multi-Hop Evaluation**: Evaluate complex queries requiring information from multiple documents
- **Comprehensive Metrics**: Precision, Recall, F1 Score, Similarity Score, and Latency
- **Flexible Configuration**: Upload custom pipeline JSON or use prebuilt configurations
- **Global Leaderboard**: Compare your results with other pipelines
- **Beautiful Visualizations**: Interactive charts and graphs for results analysis

## Project Structure

```
bench_bot2/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── routers/      # API routes
│   │   ├── services/     # Business logic (RAG pipeline, evaluator)
│   │   └── database.py   # MongoDB connection
│   └── requirements.txt
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   └── contexts/    # React contexts
│   └── package.json
└── data/                 # Dataset
    ├── corpse_data/     # Corpus files
    └── Ground_truth/    # Ground truth JSON files
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (running locally on port 27017)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory:
```
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=rag_benchmark
SECRET_KEY=your-secret-key-change-in-production
```

5. Start the backend server:
```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. **Sign Up/Login**: Create an account or login to access the platform
2. **Configure Pipeline**: 
   - Upload a custom RAG pipeline JSON file, OR
   - Use prebuilt configuration with customizable parameters
3. **Select Benchmarks**: Choose from:
   - Single Document
   - Multilingual (English, Hindi, Telugu)
   - Multi-Hop (Multiple documents)
4. **View Results**: See comprehensive metrics and visualizations
5. **Leaderboard**: Compare your results with others

## Supported Embedding Models

- `sentence-transformers/LaBSE` - Multilingual
- `setu4993/LEALLA-base` - Multilingual
- `sentence-transformers/use-cmlm-multilingual` - Multilingual
- `sentence-transformers/distiluse-base-multilingual-cased-v2` - Multilingual
- `sentence-transformers/all-MiniLM-L6-v2` - English

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Create new account
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user

### Benchmark
- `POST /api/benchmark/run` - Run benchmark evaluation
- `POST /api/benchmark/upload-pipeline` - Upload pipeline JSON
- `GET /api/benchmark/results/{benchmark_id}` - Get benchmark results

### Leaderboard
- `GET /api/leaderboard` - Get leaderboard
- `GET /api/leaderboard/{benchmark_id}` - Get benchmark details

## Evaluation Metrics

- **Precision**: Accuracy of retrieved documents
- **Recall**: Completeness of retrieval
- **F1 Score**: Harmonic mean of precision and recall
- **Similarity Score**: Semantic similarity between retrieved and ground truth
- **Latency**: Query response time in milliseconds

## Technologies Used

### Backend
- FastAPI
- MongoDB (Motor)
- Sentence Transformers
- ChromaDB / FAISS
- scikit-learn

### Frontend
- React
- Vite
- Tailwind CSS
- Framer Motion
- Recharts
- Axios

## Notes

- Models are downloaded from Hugging Face on first use and cached for faster subsequent runs
- Ensure MongoDB is running before starting the backend
- The corpus data should be in the `data/corpse_data` directory
- Ground truth files should be in the `data/Ground_truth` directory

## License

This project is for evaluation and research purposes.

