# AI Research Agent API Documentation

A comprehensive FastAPI application that provides intelligent research assistance through a modular, session-based workflow. The API leverages Google's Gemini LLM and multiple academic databases to generate research questions, analyze data requirements, identify gaps, and search relevant literature.

## 🚀 Features

- **Session Management**: Persistent workflow state across multiple API calls
- **Research Question Generation**: AI-powered main and sub-question generation
- **Data Requirements Analysis**: Automatic mapping of questions to data needs
- **Gap Analysis**: Intelligent identification of missing data variables
- **Literature Search**: Multi-database academic paper discovery
- **Complete Workflow**: Single-call complete research analysis
- **Modular Architecture**: Clean, maintainable, and scalable codebase

## 🛠️ Setup Instructions

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd Research

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start the Server
```bash
# Using the main entry point
python main.py

# Or using uvicorn directly
uvicorn app.app:create_app --host localhost --port 8000 --reload --factory
```

### 4. Access Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Setup

### Quick Start
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t research-agent .

# Run container
docker run -p 8000:8000 --env-file .env research-agent
```

## 📋 API Endpoints

### System Endpoints

- **GET** `/` - Welcome message and API information
- **GET** `/health` - Health check and system status

### Session-Based Research Workflow

#### 1. Generate Questions & Create Session
**POST** `/api/generate-questions`

Creates a new research session and generates main and sub-questions.

**Request:**
```json
{
  "title": "Maternal Health in Rural Kenya",
  "description": "Research maternal health outcomes and barriers",
  "area_of_study": "Public Health",
  "geography": "Rural Kenya",
  "custom_sub_questions": [
    "What are the training levels of healthcare providers?"
  ]
}
```

**Response:**
```json
{
  "session_id": "uuid-session-id",
  "main_question": {
    "id": "uuid",
    "text": "What are the primary factors affecting maternal health outcomes in rural Kenya?",
    "question_type": "main"
  },
  "sub_questions": [
    {
      "id": "uuid",
      "text": "What are the training levels of healthcare providers?",
      "question_type": "sub",
      "parent_question_id": "main-question-uuid"
    }
  ],
  "expires_at": "2025-09-02T10:00:00Z"
}
```

#### 2. Analyze Sub-Questions
**POST** `/api/analyze-subquestions`

Maps sub-questions to data requirements and analysis approaches.

**Request:**
```json
{
  "session_id": "uuid-session-id"
}
```

**Response:**
```json
{
  "mappings": [
    {
      "sub_question_id": "uuid",
      "sub_question": "What are the training levels of healthcare providers?",
      "data_requirements": "Healthcare provider certification records, training completion rates, continuing education data",
      "analysis_approach": "Descriptive statistics, frequency analysis, cross-tabulation by region"
    }
  ]
}
```

#### 3. Identify Data Gaps
**POST** `/api/identify-data-gaps`

Identifies missing data variables and suggests sources.

**Request:**
```json
{
  "session_id": "uuid-session-id"
}
```

**Response:**
```json
{
  "data_gaps": [
    {
      "id": "uuid",
      "missing_variable": "provider_certification_levels",
      "gap_description": "Detailed certification and training level data for rural healthcare providers",
      "suggested_sources": "Ministry of Health training records, nursing council databases",
      "sub_question_id": "uuid"
    }
  ]
}
```

#### 4. Search Literature
**POST** `/api/search-literature`

Searches academic literature for each sub-question.

**Request:**
```json
{
  "session_id": "uuid-session-id"
}
```

**Response:**
```json
{
  "literature": {
    "sub_question_id": [
      {
        "id": "uuid",
        "title": "Training and Competency of Rural Healthcare Providers",
        "authors": ["Smith, J.", "Doe, A."],
        "abstract": "This study examines...",
        "year": 2023,
        "venue": "Journal of Rural Health",
        "url": "https://doi.org/...",
        "relevance": 0.89,
        "source": "semantic_scholar"
      }
    ]
  }
}
```

#### 5. Get Session Status
**GET** `/api/session/{session_id}`

Retrieves current session state and progress.

**Response:**
```json
{
  "session_id": "uuid",
  "project": {...},
  "main_question": {...},
  "sub_questions": [...],
  "mappings": [...],
  "data_gaps": [...],
  "literature": {...},
  "expires_at": "2025-09-02T10:00:00Z",
  "status": "complete"
}
```
### Utility Endpoints

#### Complete Analysis (Single Call)
**POST** `/api/complete-analysis`

Executes the entire research workflow in a single API call.

**Request:**
```json
{
  "title": "Urban Air Quality and Health",
  "description": "Impact of air pollution on respiratory health",
  "area_of_study": "Environmental Health",
  "geography": "Nairobi, Kenya"
}
```

**Response:**
```json
{
  "main_question": {...},
  "sub_questions": [...],
  "mappings": [...],
  "data_gaps": [...],
  "literature": {...}
}
```

#### Direct Literature Search
**POST** `/api/literature/search`

Search academic literature directly without a session.

**Request:**
```json
{
  "query": "climate change health africa",
  "limit": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "Climate Change and Health in Africa",
      "authors": ["Expert, A.", "Researcher, B."],
      "abstract": "This comprehensive review...",
      "year": 2024,
      "venue": "Environmental Health Perspectives",
      "url": "https://doi.org/...",
      "relevance": 0.95,
      "source": "semantic_scholar"
    }
  ],
  "total_found": 150
}
```

#### Project Templates
**GET** `/api/project-templates`

Get sample project configurations for common research areas.

**Response:**
```json
{
  "templates": [
    {
      "name": "Public Health Research",
      "title": "Health Outcomes in {Geography}",
      "description": "Research template for public health studies",
      "area_of_study": "Public Health",
      "sample_sub_questions": [
        "What are the primary health challenges?",
        "How does access to care vary by demographic?"
      ]
    }
  ]
}
```

## 💡 Usage Examples

### Python Client

```python
import requests
import json

class ResearchClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def complete_research(self, title, description, area_of_study, geography):
        """Run complete research workflow"""
        response = requests.post(f"{self.base_url}/api/complete-analysis", json={
            "title": title,
            "description": description,
            "area_of_study": area_of_study,
            "geography": geography
        })
        return response.json()
    
    def session_workflow(self, project_data):
        """Step-by-step session-based workflow"""
        # Step 1: Generate questions
        response = requests.post(f"{self.base_url}/api/generate-questions", 
                               json=project_data)
        session_data = response.json()
        session_id = session_data["session_id"]
        
        # Step 2: Analyze sub-questions
        response = requests.post(f"{self.base_url}/api/analyze-subquestions",
                               json={"session_id": session_id})
        mappings = response.json()
        
        # Step 3: Identify gaps
        response = requests.post(f"{self.base_url}/api/identify-data-gaps",
                               json={"session_id": session_id})
        gaps = response.json()
        
        # Step 4: Search literature
        response = requests.post(f"{self.base_url}/api/search-literature",
                               json={"session_id": session_id})
        literature = response.json()
        
        return {
            "session_id": session_id,
            "questions": session_data,
            "mappings": mappings,
            "gaps": gaps,
            "literature": literature
        }

# Usage
client = ResearchClient()

# Complete analysis
result = client.complete_research(
    title="Water Quality and Child Health",
    description="Impact of water quality on childhood diseases",
    area_of_study="Environmental Health",
    geography="Rural Tanzania"
)

# Session-based workflow
project = {
    "title": "Malaria Prevention Strategies",
    "description": "Effectiveness of community-based interventions",
    "area_of_study": "Public Health",
    "geography": "Sub-Saharan Africa"
}
session_result = client.session_workflow(project)
```

### JavaScript/Node.js Client

```javascript
class ResearchAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async completeAnalysis(projectData) {
        const response = await fetch(`${this.baseUrl}/api/complete-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(projectData)
        });
        return response.json();
    }
    
    async searchLiterature(query, limit = 10) {
        const response = await fetch(`${this.baseUrl}/api/literature/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit })
        });
        return response.json();
    }
}

// Usage
const api = new ResearchAPI();

const research = await api.completeAnalysis({
    title: "Nutrition and Cognitive Development",
    description: "Impact of malnutrition on child cognitive development",
    area_of_study: "Child Development",
    geography: "East Africa"
});

const literature = await api.searchLiterature("malnutrition cognitive development", 15);
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | Yes | Google Generative AI API key | - |

### Application Settings

Configure in `config/config.py`:

```python
class Settings:
    app_name: str = "AI Research Agent API"
    version: str = "1.0.0"
    debug: bool = False
    session_expiry_hours: int = 24
    default_search_limit: int = 10
    max_search_limit: int = 50
```

## 📊 Response Formats

### Error Responses

All endpoints return consistent error formats:

```json
{
  "error": "Error type",
  "message": "Detailed error description",
  "details": {
    "field": "Additional context"
  }
}
```

### Success Responses

All successful responses include:
- Relevant data in response body
- HTTP 200 status code
- Consistent field naming (snake_case)

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### API Testing
```bash
# Run comprehensive tests
python tests/test_api.py

# Start test server
python tests/start_api.py
```

## 🏗️ Architecture

### Core Components

- **FastAPI App Factory**: `app/app.py` - Application configuration and middleware
- **API Routes**: `app/api.py` - REST endpoints with session management
- **StateGraph Workflow**: `agent_graph/` - Multi-step research process
- **LLM Integration**: `config/llm_factory.py` - Gemini model configuration
- **Data Models**: `model/models.py` - Pydantic request/response schemas

### Workflow Architecture

1. **Session Creation**: Generate questions and create persistent session
2. **Data Analysis**: Map questions to requirements and analytical approaches
3. **Gap Identification**: Find missing variables and suggest sources
4. **Literature Search**: Query multiple academic databases
5. **Results Compilation**: Aggregate and return comprehensive research data

## 📚 Additional Resources

- **API Documentation**: Available at `/docs` and `/redoc`
- **Project Structure**: See main `README.md`
- **Docker Guide**: See `docker-compose.yml`
- **Test Examples**: Check `tests/` directory

---

For support or questions, please refer to the main project documentation or submit an issue.
{
  "query": "maternal mortality Kenya",
  "limit": 10
}
```

**Response:**
```json
[
  {
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Abstract text...",
    "year": 2023,
    "venue": "Journal Name",
    "url": "https://...",
    "relevance": 0.85,
    "source": "CrossRef",
    "citations": 45
  }
]
```

### Utility Endpoints

#### Project Templates
- **GET** `/api/project-templates`
- Returns sample project templates for different research areas

**Response:**
```json
{
  "maternal_health": {
    "title": "Maternal Mortality Trends in Rural Kenya",
    "description": "Investigate the main causes of maternal deaths",
    "area_of_study": "Public Health",
    "geography": "Rural Kenya"
  },
  "urban_health": {
    "title": "Healthcare Access in Urban Slums",
    "description": "Analyze barriers to healthcare access",
    "area_of_study": "Urban Health",
    "geography": "Nairobi, Kenya"
  }
}
```

## Testing

Run the test client to verify all endpoints:
```bash
python test_api.py
```

This will test all endpoints with sample data and display the results.

## Example Usage

### Python Client Example
```python
import requests

# Generate research questions
project_data = {
    "title": "Healthcare Access in Urban Areas",
    "description": "Study barriers to healthcare access",
    "area_of_study": "Public Health",
    "geography": "Nairobi"
}

response = requests.post(
    "http://localhost:8000/api/generate-questions", 
    json=project_data
)

result = response.json()
print("Main Question:", result["main_question"]["text"])
for sq in result["sub_questions"]:
    print("Sub-question:", sq["text"])
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

const projectData = {
    title: "Healthcare Access in Urban Areas",
    description: "Study barriers to healthcare access",
    area_of_study: "Public Health",
    geography: "Nairobi"
};

axios.post('http://localhost:8000/api/generate-questions', projectData)
    .then(response => {
        console.log('Main Question:', response.data.main_question.text);
        response.data.sub_questions.forEach(sq => {
            console.log('Sub-question:', sq.text);
        });
    })
    .catch(error => {
        console.error('Error:', error.response.data);
    });
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/generate-questions" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Healthcare Access in Urban Areas",
       "description": "Study barriers to healthcare access",
       "area_of_study": "Public Health",
       "geography": "Nairobi"
     }'
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- **200**: Success
- **422**: Validation Error (invalid request data)
- **500**: Internal Server Error

Error responses include detailed error messages:
```json
{
  "detail": "Error description here"
}
```

## Rate Limiting and Performance

- The API uses caching for the sentence transformer model
- Literature search is limited to 50 results per request
- Consider implementing rate limiting for production use
- The complete analysis endpoint may take longer as it runs all workflow steps

## Production Deployment

For production deployment:

1. Set up proper environment variables
2. Use a production WSGI server (gunicorn)
3. Implement rate limiting
4. Add authentication if needed
5. Set up monitoring and logging
6. Use a reverse proxy (nginx)

Example production command:
```bash
gunicorn fastapi_main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
