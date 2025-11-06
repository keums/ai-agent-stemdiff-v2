#!/bin/bash

# Environment setup script for AI Agent Stem Diff
# This script helps set up environment variables on the EC2 instance

EC2_HOST="54.180.99.196"
EC2_USER="ubuntu"
SSH_KEY="/Users/khlk/keys/khlk-neutune2.pem"
REMOTE_DIR="/home/ubuntu/ai-agent-stemdiff"

echo "=== AI Agent Stem Diff Environment Setup ==="
echo ""
echo "This script will help you set up environment variables on your EC2 instance."
echo "Please provide the following information:"
echo ""

# Function to get user input
get_input() {
    local prompt="$1"
    local var_name="$2"
    local default_value="$3"
    
    if [ -n "$default_value" ]; then
        echo -n "$prompt [$default_value]: "
    else
        echo -n "$prompt: "
    fi
    
    read -r value
    if [ -z "$value" ] && [ -n "$default_value" ]; then
        value="$default_value"
    fi
    echo "$value"
}

# Get environment variables from user
OPENAI_API_KEY=$(get_input "Enter your OpenAI API Key" "OPENAI_API_KEY")
ANTHROPIC_API_KEY=$(get_input "Enter your Anthropic API Key" "ANTHROPIC_API_KEY")
LLM_PROVIDER=$(get_input "Enter LLM Provider (openai/claude)" "LLM_PROVIDER" "openai")
ELASTIC_CLOUD_ID=$(get_input "Enter your Elastic Cloud ID" "ELASTIC_CLOUD_ID")
ELASTIC_PASSWORD=$(get_input "Enter your Elastic Password" "ELASTIC_PASSWORD")
ES_BLOCK_INDEX=$(get_input "Enter ES Block Index" "ES_BLOCK_INDEX")
ES_ENVIRONMENTAL_SOUND_INDEX=$(get_input "Enter ES Environmental Sound Index" "ES_ENVIRONMENTAL_SOUND_INDEX")
S3_BUCKET_NAME=$(get_input "Enter S3 Bucket Name" "S3_BUCKET_NAME" "mixaudio-assets")
ROOT_BLOCK_OBJECT_URI=$(get_input "Enter Root Block Object URI" "ROOT_BLOCK_OBJECT_URI")
MUSIC_TEXT_EMBEDDING_API_URL=$(get_input "Enter Music Text Embedding API URL" "MUSIC_TEXT_EMBEDDING_API_URL")
TEXT_MUSIC_EMBEDDING_API_URL=$(get_input "Enter Text Music Embedding API URL" "TEXT_MUSIC_EMBEDDING_API_URL")
TRITON_HOST=$(get_input "Enter Triton Host" "TRITON_HOST")

echo ""
echo "Creating .env file on the server..."

# Create .env file content
cat > /tmp/ai-agent.env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=$OPENAI_API_KEY

# Anthropic API Configuration
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# LLM Provider (openai or claude)
LLM_PROVIDER=$LLM_PROVIDER

# Elasticsearch Configuration
ELASTIC_CLOUD_ID=$ELASTIC_CLOUD_ID
ELASTIC_PASSWORD=$ELASTIC_PASSWORD
ES_BLOCK_INDEX=$ES_BLOCK_INDEX
ES_ENVIRONMENTAL_SOUND_INDEX=$ES_ENVIRONMENTAL_SOUND_INDEX

# S3 Configuration
S3_BUCKET_NAME=$S3_BUCKET_NAME
ROOT_BLOCK_OBJECT_URI=$ROOT_BLOCK_OBJECT_URI

# Embedding API URLs
MUSIC_TEXT_EMBEDDING_API_URL=$MUSIC_TEXT_EMBEDDING_API_URL
TEXT_MUSIC_EMBEDDING_API_URL=$TEXT_MUSIC_EMBEDDING_API_URL

# Triton Configuration
TRITON_HOST=$TRITON_HOST
EOF

# Copy .env file to server
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/ai-agent.env "$EC2_USER@$EC2_HOST:/tmp/.env"

# Move .env file to correct location on server
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "mv /tmp/.env $REMOTE_DIR/.env"

# Set proper permissions
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "chmod 600 $REMOTE_DIR/.env"

# Restart the API service
echo "Restarting API service..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "sudo systemctl restart ai-agent-api"

# Check service status
echo "Checking service status..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "sudo systemctl status ai-agent-api --no-pager"

# Cleanup
rm -f /tmp/ai-agent.env

echo ""
echo "Environment setup completed!"
echo "You can check the API logs with:"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'sudo journalctl -u ai-agent-api -f'" 