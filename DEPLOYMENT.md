# Deployment Guide for RAG AI Chatbot

## Quick Deploy Options

### 1. Local Development (Already Working)
```bash
# Start Ollama
ollama serve

# Install dependencies
pip install -r requirements.txt

# Run backend
cd backend && python main.py

# Open frontend
open frontend/index.html
```

### 2. Docker Deployment (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Pull Ollama model (in separate terminal)
docker exec -it <ollama_container_id> ollama pull mistral
```

### 3. Railway Deployment
1. Connect GitHub repository to Railway
2. Set environment variables:
   - `OLLAMA_URL=http://localhost:11434`
   - `OLLAMA_MODEL=mistral`
3. Deploy automatically on push

### 4. Render Deployment
1. Create new Web Service on Render
2. Connect GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python backend/main.py`
5. Add environment variables

### 5. Heroku Deployment
```bash
# Install Heroku CLI
heroku create rag-ai-chatbot

# Set environment variables
heroku config:set OLLAMA_URL=http://localhost:11434
heroku config:set OLLAMA_MODEL=mistral

# Deploy
git push heroku main
```

## Production Considerations

### Security
- Add authentication middleware
- Use HTTPS
- Implement rate limiting
- Sanitize user inputs

### Performance
- Use Redis for caching
- Implement connection pooling
- Add CDN for static files
- Use load balancer

### Monitoring
- Add health checks
- Implement logging
- Monitor resource usage
- Set up alerts

### Scaling
- Horizontal scaling with Kubernetes
- Database replication
- Content delivery network
- Auto-scaling policies

## Environment Variables

### Required for Production
```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=100
HOST=0.0.0.0
PORT=8000
```

### Optional
```env
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourdomain.com
MAX_FILE_SIZE=50MB
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Railway
      uses: railway-app/railway-action@v1
```

## Troubleshooting

### Common Issues
1. **Ollama Connection Failed**
   - Ensure Ollama is running
   - Check firewall settings
   - Verify URL configuration

2. **Memory Issues**
   - Reduce chunk size
   - Use smaller embedding models
   - Add swap space

3. **Slow Performance**
   - Use GPU acceleration
   - Implement caching
   - Optimize chunking strategy

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Ollama health
curl http://localhost:11434/api/tags
```

## Monitoring Setup

### Prometheus + Grafana
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-chatbot'
    static_configs:
      - targets: ['localhost:8000']
```

### Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## Backup Strategy

### Data Backup
- Regular database dumps
- Model checkpoint backups
- Configuration backups

### Disaster Recovery
- Multi-region deployment
- Automated failover
- Data replication

## Cost Optimization

### Resource Management
- Right-size instances
- Use spot instances
- Implement auto-scaling

### Storage Optimization
- Compress embeddings
- Use efficient data structures
- Regular cleanup

---

**Ready for production deployment! 🚀**
