#!/bin/bash

# Fix Nginx Timeout Settings for AI Agent Stem Diff
# This script updates Nginx configuration to handle longer response times

EC2_HOST="54.180.99.196"
EC2_USER="ubuntu"
SSH_KEY="/Users/khlk/keys/khlk-neutune2.pem"

echo "=== Fixing Nginx Timeout Settings ==="

# Create updated Nginx configuration with longer timeouts
cat > /tmp/nginx-config-fixed << 'EOF'
server {
    listen 80;
    server_name 54.180.99.196;

    # Increase timeout settings
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    send_timeout 300s;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Additional timeout settings for frontend
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Extended timeout settings for API
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Buffer settings for large responses
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Quick timeout for health checks
        proxy_connect_timeout 10s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
EOF

echo "Copying updated Nginx configuration to server..."

# Copy the updated configuration to server
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/nginx-config-fixed "$EC2_USER@$EC2_HOST:/tmp/"

# Apply the new configuration
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "
    sudo mv /tmp/nginx-config-fixed /etc/nginx/sites-available/ai-agent
    sudo nginx -t
    sudo systemctl reload nginx
    sudo systemctl status nginx --no-pager
"

# Cleanup
rm -f /tmp/nginx-config-fixed

echo ""
echo "Nginx timeout settings updated!"
echo "Timeout values increased to 300-600 seconds for API calls."
echo ""
echo "You can check Nginx logs with:"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'sudo tail -f /var/log/nginx/error.log'"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_HOST 'sudo tail -f /var/log/nginx/access.log'" 