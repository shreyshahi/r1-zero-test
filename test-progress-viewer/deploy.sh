#!/bin/bash

# Build the React app
npm run build

# Sync the build directory with S3
aws s3 sync build/ s3://rl-test-set-progress-viewer --delete

# Sync the data directory with S3
aws s3 sync public/data/ s3://rl-test-set-progress-viewer/data --delete

# Set cache control for static assets
aws s3 cp s3://rl-test-set-progress-viewer/static/ s3://rl-test-set-progress-viewer/static/ --recursive --metadata-directive REPLACE --cache-control 'max-age=31536000' --acl public-read

# Set cache control for HTML and JSON files
aws s3 cp s3://rl-test-set-progress-viewer/index.html s3://rl-test-set-progress-viewer/index.html --metadata-directive REPLACE --cache-control 'no-cache' --acl public-read
aws s3 cp s3://rl-test-set-progress-viewer/data/ s3://rl-test-set-progress-viewer/data/ --recursive --metadata-directive REPLACE --cache-control 'no-cache' --acl public-read
