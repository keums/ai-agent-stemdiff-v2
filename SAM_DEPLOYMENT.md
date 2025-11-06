# AWS SAM Deployment Guide

## 📦 AWS SAM Architecture

This project uses AWS SAM (Serverless Application Model) to deploy a serverless API with the following components:

### 🏗️ Infrastructure Components

- **API Gateway**: REST API endpoints
- **WebSocket API**: Real-time communication
- **Lambda Functions**: Serverless compute
- **DynamoDB**: WebSocket connection storage
- **IAM Roles**: Security and permissions

### 🔧 Lambda Functions

| Function | Purpose | Endpoint |
|----------|---------|----------|
| `HealthFunction` | Health check | `GET /health` |
| `GenerateStemDiffFunction` | Generate stem differences | `POST /generate-stem-diff` |
| `MusicAgentFunction` | Music analysis | `POST /music-agent` |
| `WebSocketConnectFunction` | WebSocket connection | `$connect` |
| `WebSocketDisconnectFunction` | WebSocket disconnection | `$disconnect` |
| `WebSocketDefaultFunction` | WebSocket message handling | `$default` |

## 🚀 Prerequisites

### 1. Install Required Tools

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Install SAM CLI
pip install aws-sam-cli

# Install Docker (for local testing)
# Download from https://www.docker.com/products/docker-desktop
```

### 2. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region, and Output format
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## 🏠 Local Development

### Start Local API Gateway

```bash
# Start local API server
./local-sam.sh api

# API will be available at:
# http://localhost:3000/health
# http://localhost:3000/generate-stem-diff
# http://localhost:3000/music-agent
```

### Test Individual Functions

```bash
# Invoke specific function
./local-sam.sh invoke HealthFunction
./local-sam.sh invoke GenerateStemDiffFunction
```

### Local Testing Examples

```bash
# Test health endpoint
curl http://localhost:3000/health

# Test generate stem diff
curl -X POST http://localhost:3000/generate-stem-diff \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Add more bass",
    "context_song_info": {"bpm": 120},
    "generated_stem_diff_uris": [],
    "mix_stem_diff_info": []
  }'

# Test music agent
curl -X POST http://localhost:3000/music-agent \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a upbeat electronic track",
    "context": {}
  }'
```

## 🚀 Production Deployment

### Deploy to AWS

```bash
# Deploy to development environment
./deploy-sam.sh dev

# Deploy to production environment  
./deploy-sam.sh prod
```

### Manual Deployment

```bash
# Build
sam build

# Deploy with parameters
sam deploy \
  --stack-name "ai-agent-stemdiff-prod" \
  --parameter-overrides \
    Environment="prod" \
    OpenAIAPIKey="your_openai_key" \
    AnthropicAPIKey="your_anthropic_key" \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

## 🔍 Monitoring and Debugging

### View Logs

```bash
# View all logs
sam logs --stack-name ai-agent-stemdiff-dev --tail

# View specific function logs
sam logs --stack-name ai-agent-stemdiff-dev --name HealthFunction --tail
```

### CloudWatch Logs

```bash
# List log groups
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/ai-agent-stemdiff"

# View logs for specific function
aws logs tail /aws/lambda/ai-agent-stemdiff-dev-HealthFunction --follow
```

## 🌐 WebSocket Usage

### Connect to WebSocket

```javascript
const ws = new WebSocket('wss://your-websocket-api-id.execute-api.us-east-1.amazonaws.com/dev');

ws.onopen = function() {
    console.log('Connected to WebSocket');
    
    // Send ping
    ws.send(JSON.stringify({
        action: 'ping'
    }));
    
    // Send stem diff request
    ws.send(JSON.stringify({
        action: 'generate_stem_diff',
        data: {
            prompt: 'Add more drums',
            context: {}
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 📊 API Endpoints

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/generate-stem-diff` | Generate stem differences |
| POST | `/music-agent` | Music analysis and suggestions |

### WebSocket Routes

| Route | Description |
|-------|-------------|
| `$connect` | Client connection |
| `$disconnect` | Client disconnection |
| `$default` | Message handling |

## 🗑️ Cleanup

### Delete Stack

```bash
# Delete development stack
sam delete --stack-name ai-agent-stemdiff-dev

# Delete production stack
sam delete --stack-name ai-agent-stemdiff-prod
```

## 🔧 Configuration

### Environment Variables

The following environment variables are available in Lambda functions:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `LLM_PROVIDER`: LLM provider (openai/claude)
- `WEBSOCKET_API_ENDPOINT`: WebSocket API endpoint
- `AWS_SAM_STACK_NAME`: Stack name for DynamoDB table

### Customization

You can customize the deployment by modifying:

- `template.yaml`: Infrastructure configuration
- `samconfig.toml`: SAM CLI configuration
- Lambda function code in `lambda_functions/` directories

## 🐛 Troubleshooting

### Common Issues

1. **Build Fails**:
   - Check Python dependencies in `requirements.txt`
   - Ensure Docker is running for local testing

2. **Deployment Fails**:
   - Verify AWS credentials and permissions
   - Check CloudFormation events in AWS Console

3. **Lambda Function Errors**:
   - Check CloudWatch logs
   - Verify environment variables are set correctly

4. **CORS Issues**:
   - CORS headers are configured in the template
   - Check API Gateway CORS settings if needed

### Debug Commands

```bash
# Validate template
sam validate

# Check build
sam build --debug

# Test function locally
sam local invoke HealthFunction --event test-events/health.json

# Check CloudFormation stack
aws cloudformation describe-stacks --stack-name ai-agent-stemdiff-dev
```