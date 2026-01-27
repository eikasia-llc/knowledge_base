# Infrastructure Definitions and Requirements
- id: infrastructure_definitions_and_requirements
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->

## Preferred technology stack
- id: infrastructure_definitions_and_requirements.preferred_technology_stack
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
This technologies are to be preferred unless you are specifically asked otherwise, or asked for alternative analysis.

- Target Architecture:
- Frontend: React (Vite/CRA) $\rightarrow$ Firebase Hosting (Global CDN, Edge caching).
- Backend: Python (Flask/FastAPI) $\rightarrow$ Google Cloud Run (Serverless containers).
- Database: Firestore (App State/User Data) + BigQuery (Analytics/Telemetry).
- AI/Compute: Vertex AI (Model Training/Inference) + Artifact Registry (Docker Images).

## Cost Policy
- id: infrastructure_definitions_and_requirements.cost_policy
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
- Seek low cost alternatives.
- Always prefer alternatives that have a low cost from the start over alternatives that have a higher base cost but promise to eventually be low in the long run or when scaled.
- Be cheap rather than elegant.

## Requirements & safety when designing or provisioning infraestructure
- id: infrastructure_definitions_and_requirements.requirements_safety_when_designing_or_provisioning_infraestructure
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
Pre existing infraestructure may not follow this guidelines because of conditions taken into account. If that is the case, the exception will be documented. If you encounter undocumented exceptions in pre existing infrastructure, generate a warining in your worklog and continue. Take this into account when designing, updating or provisioning infraestructure:

### Interoperability
- id: infrastructure_definitions_and_requirements.requirements_safety_when_designing_or_provisioning_infraestructure.interoperability
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
- Soft Multi-tenancy preferred over GCP Project separation.
- Always enable Private Google Access in subnets. It allows your Cloud Run instances to talk to Google APIs over Google's internal network for free.
- If you provision a new GKE cluster, make sure to grant permission to GCP Projects so the can deploy to it.
- If you provision a new GCP Project, make sure it can comunicate with other projects via a Shared VPC newtork and that the can deploy to all GKE clusters.
- Detect if the new solution will require changes in the customer side. If so, ask for permission before continuing.

### Cost Safety
- id: infrastructure_definitions_and_requirements.requirements_safety_when_designing_or_provisioning_infraestructure.cost_safety
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
- Unless specified otherwise, all deployments will configure logging severity to WARN (warning and error).
- Beware of Logging Explosion risks.
- Beware of infinite or recursive scaling risks.
- Detect if the new solution will use Premium Tier features instead of Standard Tier (Regional only). If so, ask for permission before continuing.
- Detect if the new solution will have ingestion into Cloud Logging, warn the human user, ask for permission before continuing.
- Detect if the new solution will have data moving between data centers or regions, warn the human user, ask for permission before continuing.
- Detect if the new solution will have increased internet traffic, warn the human user, ask for permission before continuing.
- All new resources are to be provisioned in `us-central1` (Iowa) region for cheaper TPUs/GPUs, storage and network.

### Rollback Safety
- id: infrastructure_definitions_and_requirements.requirements_safety_when_designing_or_provisioning_infraestructure.rollback_safety
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
- Detect if the new solution will delete data or configuration that cannot be easily undone. If so, ask for permission before continuing.
