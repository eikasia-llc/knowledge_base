# Cloud Infrastructure: Knowledge Base

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
| **Compute** | **Google Cloud Run** | Hosts the Streamlit application as a serverless container. Scalable and only runs when requests are active. |
| **Authentication** | **Identity-Aware Proxy (IAP)** | Secures the `run.app` URL. Intercepts incoming requests and requires a Google login before granting access to authorized users. |
| **Secret Management** | **Secret Manager** | Securely stores the GitHub Personal Access Token (`GITHUB_TOKEN`) used for repository synchronization. |
| **Persistence** | **Git / GitHub** | Acts as the source of truth for knowledge files. The app pulls content from GitHub to ephemeral storage on startup. |
| **Registry** | **Artifact Registry / GCR** | Stores the Docker container images for the application. |
| **Storage (Ephemeral)** | **In-memory / `/tmp`** | Stores the cloned repository during the container's lifetime. Note that this is cleared when the instance shuts down. |

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
