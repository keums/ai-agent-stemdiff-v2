#!/bin/bash

# AWS SAM Local Development Script

set -e

echo "🏠 === AWS SAM Local Development ==="
echo ""

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "❌ AWS SAM CLI is not installed. Please install it first."
    echo "Install with: pip install aws-sam-cli"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed!"
echo ""

# Build the application
echo "🔨 Building SAM application..."
sam build

# Create env.json for local testing
cat > env.json << EOF
{
  "HealthFunction": {
    "Environment": "local"
  },
  "GenerateStemDiffFunction": {
    "Environment": "local",
    "OPENAI_API_KEY": "${OPENAI_API_KEY:-your_openai_api_key_here}",
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-your_anthropic_api_key_here}",
    "LLM_PROVIDER": "openai"
  },
  "MusicAgentFunction": {
    "Environment": "local",
    "OPENAI_API_KEY": "${OPENAI_API_KEY:-your_openai_api_key_here}",
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-your_anthropic_api_key_here}",
    "LLM_PROVIDER": "openai"
  }
}
EOF

echo "📝 Created env.json for local environment variables"
echo ""

# Determine which service to start
SERVICE=${1:-api}

case $SERVICE in
    "api")
        echo "🌐 Starting SAM Local API Gateway on http://localhost:3000..."
        echo "📖 API Documentation will be available at:"
        echo "   - Health: http://localhost:3000/health"
        echo "   - Generate Stem Diff: http://localhost:3000/generate-stem-diff"
        echo "   - Music Agent: http://localhost:3000/music-agent"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        sam local start-api --env-vars env.json --host 0.0.0.0 --port 3000
        ;;
    
    "lambda")
        echo "🔧 Starting SAM Local Lambda service..."
        sam local start-lambda --env-vars env.json
        ;;
    
    "invoke")
        FUNCTION=${2:-HealthFunction}
        echo "📞 Invoking function: $FUNCTION"
        sam local invoke "$FUNCTION" --env-vars env.json
        ;;
    
    *)
        echo "❌ Unknown service: $SERVICE"
        echo ""
        echo "Usage: $0 [api|lambda|invoke] [function_name]"
        echo ""
        echo "Examples:"
        echo "  $0 api                    # Start API Gateway"
        echo "  $0 lambda                 # Start Lambda service"
        echo "  $0 invoke HealthFunction  # Invoke specific function"
        exit 1
        ;;
esac