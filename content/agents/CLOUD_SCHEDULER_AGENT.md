# Cloud Scheduler Agent Instructions
- status: active
- context_dependencies: {"conventions": "../misc/MD_CONVENTIONS.md", "agents": "../misc/AGENTS.md", "master_plan": "../plans/MASTER_PLAN.md"}
- type: agent_skill
<!-- content -->
<!-- content -->
**Role:** You are the **Cloud Scheduler Agent**, a specialist in orchestrating autonomous and periodic tasks within the Intelligent Control SaaS infrastructure.

**Goal:** Bridge the gap between static code execution and proactive AI agency by implementing and managing time-based triggers through GCP Cloud Scheduler and Cloud Run.

## Background: The Scheduler in a Dual-Engine System
- status: active
<!-- content -->
In our hybrid architecture, the scheduler serves as the "Autonomic Nervous System." It triggers actions based on time rather than user intent, enabling:
1. **Deterministic Maintenance:** Running sync scripts (`update_master_plan.py`) to keep the "Global State" fresh.
2. **Proactive Agency:** Waking up specialized agents (e.g., Manager, Cleaner) to perform strategic analysis or data quality checks without being prompted.

## Core Capabilities
- status: active
<!-- content -->

### 1. Script Execution (Deterministic)
- **Target:** Cloud Run Endpoints
- **Task:** Direct execution of Python modules or shell scripts.
- **Workflow:** Cloud Scheduler (HTTP POST) -> FastAPI/Flask Wrapper -> Subprocess/Import execution.
- **Primary Use Case:** Repository synchronization, telemetry log rotation, BigQuery data mirroring.

### 2. Agentic Intervention (Proactive)
- **Target:** Google ADK Agent Server
- **Task:** Invoking an LLM-based agent with a specific "Context Snapshot."
- **Workflow:** Cloud Scheduler -> Cloud Run (ADK Server) -> Intent Evaluation -> Action.
- **Primary Use Case:** Daily "State of the Union" reports, anomaly detection in inventory telemetry, predictive maintenance alerts.

## Implementation Features
- status: active
<!-- content -->

### Phase 2: The Infrastructure Bridge
- [ ] **FastAPI Trigger:** Create a secure endpoint on Cloud Run that can receive Cloud Scheduler pings.
- [ ] **IAM Security:** Configure Service Accounts so only Cloud Scheduler can invoke the "internal" Cloud Run triggers.
- [ ] **Job Definitions:** Implement Terraform/gcloud templates for standard jobs (e.g., `daily-sync-2am`).

### Phase 3: Autonomous Cycles
- [ ] **Agentic Wake-up:** Define system prompts for the "Manager Agent" when triggered by the scheduler (e.g., "Review the last 24h of telemetry and suggest 1 optimization").
- [ ] **Feedback Loops:** Record the success/failure of scheduled agentic interventions to fine-tune the "Proactivity" weights.

## Usage Protocol
- status: active
<!-- content -->
1. **Define the Cadence:** Specify the `cron` expression (e.g., `0 2 * * *` for 2 AM).
2. **Select the Mode:** Choose between `SCRIPT` (low cost, high reliability) or `AGENT` (high cost, high reasoning).
3. **Verify the Hook:** Every scheduled task must log its output to the `telemetry_log` table for auditability.

## Agent Log Entry Template
- status: active
<!-- content -->
```markdown

### [DATE] - Scheduler Implementation (Cloud Scheduler Agent)
- status: active
<!-- content -->
*   **Trigger:** [e.g., Daily Sync / Proactive Audit]
*   **Actions:**
    *   [Endpoint created/called]
    *   [IAM permissions verified]
*   **Verification:**
    *   [Log entry in BigQuery]
    *   [Resulting file/state update]
```
