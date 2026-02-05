#!/bin/bash

## deploy.sh is a straightforward local docker build → docker push → gcloud run deploy.

# Configuration
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="knowledge-base-app"
REGION="us-central1"
# We are using Google Artifact Registry
#IMAGE_NAME="knowledge-base-repo/${PROJECT_ID}/${SERVICE_NAME}" # old?
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/knowledge-base-repo/${SERVICE_NAME}"


# Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}..."
docker build -t ${IMAGE_NAME} .

# Push to Artifact Registry/GCR
echo "Pushing image to GCR..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --set-env-vars="GITHUB_REPO_URL=https://github.com/eikasia-llc/knowledge_base.git" \
    --update-secrets="GITHUB_TOKEN=GITHUB_TOKEN:latest" \
    --memory 1Gi \
    --cpu 1

echo "Deployment complete!"
