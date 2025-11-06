#!/bin/bash

# Update script for AI Agent Stem Diff application
# This script updates the code and restarts services without full redeployment

# Error handling - don't exit on error, but track failures
set -o pipefail  # Exit on pipe failures
ERRORS=0

# Function to handle errors
handle_error() {
    local exit_code=$?
    print_error "Command failed with exit code $exit_code"
    ERRORS=$((ERRORS + 1))
    return $exit_code
}

# Set up error handling
trap handle_error ERR

# Configuration
EC2_HOST="54.180.99.196"
EC2_USER="ubuntu"
SSH_KEY="~/keys/khlk-neutune2.pem"
PROJECT_NAME="ai-agent-stemdiff"
REMOTE_DIR="/home/ubuntu/$PROJECT_NAME"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to execute remote command
remote_exec() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "$1"
}

# Function to copy files to remote
copy_to_remote() {
    rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='output/' \
        --exclude='.DS_Store' \
        ./ "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"
}

print_status "Starting update process..."

# Step 1: Copy updated files
print_status "Copying updated files to remote server..."
copy_to_remote

# Step 2: Update Python dependencies if requirements.txt changed
print_status "Updating Python dependencies..."
remote_exec "cd $REMOTE_DIR && source venv/bin/activate && pip install -r requirements.txt"

# Step 3: Rebuild frontend for production
print_status "Rebuilding frontend for production..."
remote_exec "cd $REMOTE_DIR/frontend && npm install"
remote_exec "cd $REMOTE_DIR/frontend && npm run build:prod"

# Step 4: Restart services
print_status "Restarting services..."
remote_exec "sudo systemctl restart ai-agent-api"
remote_exec "sudo systemctl restart ai-agent-frontend"

# Step 5: Check service status
print_status "Checking service status..."
remote_exec "sudo systemctl status ai-agent-api --no-pager"
remote_exec "sudo systemctl status ai-agent-frontend --no-pager"

# Final summary
echo ""
if [ $ERRORS -eq 0 ]; then
    print_status "Update completed successfully with no errors!"
else
    print_error "Update completed with $ERRORS error(s). Please check the logs above."
    echo ""
    echo "=== Troubleshooting Tips ==="
    echo "1. Check if the services are running: sudo systemctl status ai-agent-api"
    echo "2. View logs: sudo journalctl -u ai-agent-api -f"
    echo "3. Try restarting services manually"
    echo ""
fi

echo "=== Update Information ==="
echo "Frontend URL: http://$EC2_HOST"
echo "API URL: http://$EC2_HOST/api"
echo "Health Check: http://$EC2_HOST/health"
echo "" 