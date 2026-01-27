# control_tower
- id: control_tower
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->

## Project Overview
- id: control_tower.project_overview
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
Home Base for the Infrastructure Agent

## Structure
- id: control_tower.structure
- status: active
- type: context
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "AGENTS.md", "project_root": "README.md"}
- last_checked: 2026-01-25
<!-- content -->
At the root it has markdown files to be used as context (aka "system promt"), as well as files describing the company's plans and infraestructure (current or desired). It also holds the artifacts created by the aagent such as work logs, screenshots, documents and other. 

The `repositories` directory is for the agent to find and clone all other git repositories with apps or code to be deployed, as well as IAC repositories. That directory is git ignored and the agent can safely interact with the repositories inside there. 

The `artifacts` directory has all the artifacts created by the agent such as work logs, screenshots, documents and other.

## Key Files
- status: active
<!-- content -->
- **`INFRASTRUCTURE_AGENT.md`**: The entry point for the Infrastructure Agent.
- **`INFRASTRUCTURE_PLAN.md`**: Evolving High Level Plan for the Infrastructure.
- **`INFRASTRUCTURE_DEFINITIONS.md`**: Describes company policy & guidelines on infraestructure and its evolution.
- **`AGENTS_LOG.md`**: High level log of the agent's actions.
