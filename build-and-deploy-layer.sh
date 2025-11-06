#!/bin/bash

# Quick script to build and deploy the pysoundfile layer
set -e

echo "Building and deploying pysoundfile Lambda layer..."
echo "=================================================="

# Navigate to layer directory
cd layers/pysoundfile-layer

echo "Step 1: Building the layer with Docker..."
bash build-layer.sh

echo ""
echo "Step 2: Deploying the layer to AWS..."
bash deploy-layer.sh

echo ""
echo "Step 3: The layer is now ready!"
echo "Your SAM template has been updated to use the new layer."
echo "You can now deploy your SAM application with:"
echo "  sam deploy"
echo ""
echo "Layer files created in:"
echo "  layers/pysoundfile-layer/"
