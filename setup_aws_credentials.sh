#!/bin/bash

# AWS Credentials Setup Script for AI Agent Stem Diff
# This script sets up AWS credentials on the EC2 instance

EC2_HOST="54.180.99.196"
EC2_USER="ubuntu"
SSH_KEY="/Users/khlk/keys/khlk-neutune2.pem"

echo "=== AWS Credentials Setup ==="
echo ""
echo "This script will help you set up AWS credentials on your EC2 instance."
echo "Please provide your AWS credentials:"
echo ""

# Function to get user input
get_input() {
    local prompt="$1"
    local var_name="$2"
    
    echo -n "$prompt: "
    read -r value
    echo "$value"
}

# Get AWS credentials from user
AWS_ACCESS_KEY_ID=$(get_input "Enter your AWS Access Key ID" "AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY=$(get_input "Enter your AWS Secret Access Key" "AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION=$(get_input "Enter your AWS Default Region (e.g., us-east-1)" "AWS_DEFAULT_REGION" "us-east-1")

echo ""
echo "Setting up AWS credentials on the server..."

# Create AWS credentials directory and files on server
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "mkdir -p ~/.aws"

# Create credentials file content
cat > /tmp/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF

# Create config file content
cat > /tmp/config << EOF
[default]
region = $AWS_DEFAULT_REGION
output = json
EOF

# Copy files to server
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/credentials "$EC2_USER@$EC2_HOST:~/.aws/"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/config "$EC2_USER@$EC2_HOST:~/.aws/"

# Set proper permissions
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "chmod 600 ~/.aws/credentials ~/.aws/config"

# Test AWS credentials
echo "Testing AWS credentials..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "aws sts get-caller-identity"

# Restart the API service
echo "Restarting API service..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "sudo systemctl restart ai-agent-api"

# Check service status
echo "Checking service status..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "sudo systemctl status ai-agent-api --no-pager"

# Cleanup
rm -f /tmp/credentials /tmp/config

echo ""
echo "AWS credentials setup completed!"
echo "You can check the API logs with:"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'sudo journalctl -u ai-agent-api -f'" 