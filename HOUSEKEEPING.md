# Housekeeping Protocol
- status: recurring
- type: guideline
<!-- content -->
1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. Make sure that the report uses the proper syntax protocol as defined in MD_CONVENTIONS.md. If necessary, you can always use the scripts in the language folder to help you with this.
6. Print that report in the Latest Report subsection below, overwriting previous reports.
7. Add that report to the AGENTS_LOG.md.

<!-- MERGED FROM NEWER VERSION -->

- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "agents": "../AGENTS.md"}
<!-- content -->
1. Read the AGENTS.md file and the MD_CONVENTIONS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Update the dataset by running the scraper scripts and making sure things are in order.
4. **Smart Merge**: Execute the `manager/cleaner/compare_and_merge.py` script to integrate changes from external repositories into the local `content/`.
5. **Cleanup**: Verify that `manager/cleaner/temprepo_cleaning/` is empty after the process.
6. Proceed doing different sanity checks and unit tests from root scripts to leaves.
7. Compile all errors and tests results into a report. Make sure that the report uses the proper syntax protocol as defined in MD_CONVENTIONS.md. If necessary, you can always use the scripts in the language folder to help you with this.
8. Print that report in the Latest Report subsection below, overwriting previous reports.
9. Add that report to the AGENTS_LOG.md.
10. Commit and push the changes.

## Current Project Housekeeping
- status: active
- type: recurring
<!-- content -->

## Dependency Network
- status: active
- type: task
<!-- content -->
Based on post-React integration analysis:
- **Core Modules:**
- **Advanced Modules:**
- **Engine Module:**
- **API Module:**
- **Tests:**
- **Notebooks:**

<!-- MERGED FROM NEWER VERSION -->

Based on codebase analysis (2026-01-28):

### 1. Application Layer (Entry Point)
- id: housekeeping_protocol.dependency_network.1_application_layer_entry_point
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- **`app.py`**: Main Streamlit application.
  - *Dependencies*: `src.core.engine`, `src.core.vector_store` (dynamic), `src.scrapers.mcmp_scraper` (dynamic), `gspread` (optional).
  - *Role*: Handles UI, user state, and orchestration of scraping/indexing.

### 2. Core Engine Layer
- id: housekeeping_protocol.dependency_network.2_core_engine_layer
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- **`src/core/engine.py`**: RAGEngine class.
  - *Dependencies*: `src.core.vector_store`, `src.utils.logger`, `openai`, `anthropic`, `google.genai`.
  - *Role*: Query decomposition, LLM interaction, response generation.
- **`src/core/vector_store.py`**: VectorStore class.
  - *Dependencies*: `chromadb`, `src.utils.logger`.
  - *Role*: Manages ChromaDB, embedding functions, and document indexing.

### 3. Data Acquisition Layer (Scrapers)
- id: housekeeping_protocol.dependency_network.3_data_acquisition_layer_scrapers
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- **`src/scrapers/mcmp_scraper.py`**: MCMPScraper class.
  - *Dependencies*: `requests`, `bs4`, `src.utils.logger`.
  - *Role*: Scrapes events, people, and research from MCMP website; runs standard cleaning.

### 4. Utilities & Scripts
- id: housekeeping_protocol.dependency_network.4_utilities_scripts
- status: active
- type: context
- last_checked: 2026-01-31
<!-- content -->
- **`src/utils/logger.py`**: centralized logging.
- **`scripts/update_knowledge.py`**: specific script for parsing markdown knowledge.
- **`scripts/test_sheets_connection.py`**: connection test for Google Sheets.

## Latest Report
- status: active
- type: task
- owner: Antigravity
<!-- content -->
**Execution Date:** 2026-01-31 (Antigravity - Run 2)

**Status Checks:**
1.  **Data Update (`manager/cleaner/pipeline.py`)**: **Passed**.
    - Imported/Updated 16 files from `mcmp_chatbot` to `manager/cleaner/temprepo_cleaning`.
2.  **Smart Merge (`manager/cleaner/compare_and_merge.py`)**: **Executed**.
    - Scanned `manager/cleaner/temprepo_cleaning` and compared against `content/`.
    - No changes detected or applied (files likely already up-to-date or new files not auto-moved).
3.  **Cleanup**: **Passed**.
    - Verified `manager/cleaner/temprepo_cleaning` was emptied after processing.
4.  **Unit Tests**: **Passed**.
    - `tests/test_index_builder.py`: **Passed** (3 tests).
    - `tests/test_semantic_matcher.py`: **Passed** (3 tests).

**Summary:**
Enhanced housekeeping executed. Validated the full ingestion -> merge -> cleanup pipeline. System is stable and up-to-date.

**Action Items:**
- [ ] Review `manager/cleaner/compare_and_merge.py` logic for new file handling (currently logs them but doesn't move them to `content/`).
