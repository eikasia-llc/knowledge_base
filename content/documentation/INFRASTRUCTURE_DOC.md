# Cloud Infrastructure: Knowledge Base
- status: active
- type: documentation

This document details the cloud components and security layers used to deploy and protect the Knowledge Base application on Google Cloud Platform.

## Architecture Diagram

```mermaid
graph TD
    User([User / Web Browser]) --> IAP[Identity-Aware Proxy]
    IAP -- Authentication --> CR[Cloud Run Service]
    CR -- Runtime Files --> Temp[/tmp/knowledge_base_repo]
    CR -- Pull/Push --> GitHub[GitHub Repository]
    CR -- Credentials --> SM[Secret Manager]
    SM -- GITHUB_TOKEN --> CR
    AR[Artifact Registry] -- Container Image --> CR
```

## Component Enumeration

| Component | Service | Purpose |
| :--- | :--- | :--- |
| **Registry** | **Artifact Registry** | Stores the Docker container images. Repo: `knowledge-base-repo` (`us-central1`). Full image path: `us-central1-docker.pkg.dev/eikasia-ops/knowledge-base-repo/knowledge-base-app`. |
| **Compute** | **Google Cloud Run** | Hosts the Streamlit application as a serverless container. Scalable and only runs when requests are active. |
| **Authentication** | **Identity-Aware Proxy (IAP)** | Secures the `run.app` URL. Intercepts incoming requests and requires a Google login before granting access to authorized users. |
| **Secret Management** | **Secret Manager** | Securely stores the GitHub Personal Access Token (`GITHUB_TOKEN`) used for repository synchronization. |
| **Persistence** | **Git / GitHub** | Acts as the source of truth for knowledge files. The app pulls content from GitHub to ephemeral storage on startup. |
| **Storage (Ephemeral)** | **In-memory / `/tmp`** | Stores the cloned repository during the container's lifetime. Note that this is cleared when the instance shuts down. |

## Build

### Create repository in Google Artifact for image push

```
gcloud artifacts repositories create knowledge-base-repo \
    --repository-format=docker \
    --location=us-central1 \
    --project=eikasia-ops \
    --description="Docker images for knowledge_base app" 2>&1
```
### Authenticate Docker for Artifact Registry (first time on a machine)

```
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Cloud Build vs Docker Build

Cloud Build runs on Google's shared infrastructure. When you run gcloud builds submit:

1. Your local source is tarball'd and uploaded to a GCS bucket (gs://<project>_cloudbuild/source/). That's the only upload from your machine.                                         
2. A ephemeral VM spins up on Google's side. It pulls that tarball, runs each step sequentially — each step is its own container (gcr.io/cloud-builders/docker, etc.).
3. Each step's container does its work (build, push, deploy) inside GCP. The final image push goes from that VM to Artifact Registry — same network, no public internet.              

Bandwidth comparison:

```
  ┌────────────────────────────────────┬────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │             What moves             │                 deploy.sh (local)                  │               cloudbuild.yaml (Cloud Build)                │
  ├────────────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Source upload                      │ nothing (already local)                            │ tarball → GCS (small, ~3.5 MiB here)                       │
  ├────────────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Base image pull (python:3.11-slim) │ your machine ← Docker Hub                          │ Cloud Build VM ← Docker Hub                                │
  ├────────────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Cache image pull (--cache-from)    │ N/A                                                │ Cloud Build VM ← Artifact Registry (same region, internal) │
  ├────────────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Final image push                   │ your machine → Artifact Registry (public internet) │ Cloud Build VM → Artifact Registry (internal)              │
  └────────────────────────────────────┴────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘
```

The big difference is the final image push. That's the heaviest artifact (the full runtime image). With deploy.sh that crosses your public internet connection. With Cloud Build it
stays inside GCP.

The base image pull from Docker Hub hits public internet in both cases. That's why --cache-from on the builder stage matters — if requirements.txt hasn't changed, the builder step
skips rebuilding entirely and the final image pull from Artifact Registry is internal and fast.

TL;DR: Cloud Build trades a small source upload for keeping the heavy image push internal to GCP. deploy.sh pushes that heavy image over your own connection.

## Deployment Commands

### Streamlit app to GCloud Run

The `deploy.sh` handles that. Read it completely to understand what it needs.

### On Ephemeral Storage

In Google Cloud Run, a container's life doesn't end the moment it finishes sending an HTTP response. Instead, it enters a state of "idle" where Google keeps it alive to avoid the performance penalty of a "cold start" for the next request.

#### The "Idle" Grace Period
Once your container sends the final byte of a response, it is marked as idle.
Default Duration: Cloud Run typically keeps idle instances alive for up to 15 minutes.
Purpose: This "warm" state allows the container to handle subsequent requests instantly. If a new request arrives during this window, the container is "reused," and the idle timer resets.
Persistent storage is github. Manually synced via UI buttons.

## Security Layers

1. **Identity-Aware Proxy (IAP)**: The first line of defense. Only users in the `eikasia.com` organization with the `IAP-secured Web App User` role can reach the application.
2. **IAM Controls**: The Cloud Run service account is restricted to minimal permissions (`Secret Manager Secret Accessor` only for specific secrets).
3. **Secret Masking**: The GitHub PAT is injected as a secret reference, never exposed in environment variables or logs in plain text.

### Permissions

To access the app: Grant access to your specific account (Recommended)
Instead of making it public, we grant ourseles permission to view it. 
This is the command:

```
gcloud run services add-iam-policy-binding knowledge-base-app \
    --region=us-central1 \
    --member="user:eikasia@eikasia.com" \
    --role="roles/run.invoker" \
    --project=eikasia-ops
```
