#!/bin/bash

# Build the React app
npm run build

# First sync everything except the special files
aws s3 sync build/ s3://rl-test-progress-viewer --delete --exclude "*.html" --exclude "*.css" --exclude "*.js" --exclude "*.json" --exclude "*.map"

# Then sync with specific content types
aws s3 sync build/ s3://rl-test-progress-viewer --delete --content-type "text/html" --exclude "*" --include "*.html"
aws s3 sync build/ s3://rl-test-progress-viewer --delete --content-type "text/css" --exclude "*" --include "*.css"
aws s3 sync build/ s3://rl-test-progress-viewer --delete --content-type "application/javascript" --exclude "*" --include "*.js"
aws s3 sync build/ s3://rl-test-progress-viewer --delete --content-type "application/json" --exclude "*" --include "*.json"
aws s3 sync build/ s3://rl-test-progress-viewer --delete --content-type "application/json" --exclude "*" --include "*.map"

# Set specific content types for images
aws s3 cp build/favicon.ico s3://rl-test-progress-viewer/favicon.ico --content-type "image/x-icon"
aws s3 cp build/logo192.png s3://rl-test-progress-viewer/logo192.png --content-type "image/png"
aws s3 cp build/logo512.png s3://rl-test-progress-viewer/logo512.png --content-type "image/png"

# Sync the data directory with S3
aws s3 sync public/data/ s3://rl-test-progress-viewer/data --delete --content-type "application/json"

# Verify files were uploaded
echo "Verifying manifest.json..."
aws s3 ls s3://rl-test-progress-viewer/data/manifest.json

echo "Listing data files..."
aws s3 ls s3://rl-test-progress-viewer/data/ | head -n 5

# Set metadata for better caching
aws s3 cp s3://rl-test-progress-viewer/static/ s3://rl-test-progress-viewer/static/ --recursive --metadata-directive REPLACE --cache-control 'max-age=31536000' --content-type "text/css" --exclude "*" --include "*.css"
aws s3 cp s3://rl-test-progress-viewer/static/ s3://rl-test-progress-viewer/static/ --recursive --metadata-directive REPLACE --cache-control 'max-age=31536000' --content-type "application/javascript" --exclude "*" --include "*.js"

# Set cache control for HTML and JSON files
aws s3 cp s3://rl-test-progress-viewer/index.html s3://rl-test-progress-viewer/index.html --metadata-directive REPLACE --cache-control 'no-cache' --content-type "text/html"
aws s3 cp s3://rl-test-progress-viewer/data/ s3://rl-test-progress-viewer/data/ --recursive --metadata-directive REPLACE --cache-control 'no-cache' --content-type "application/json"
