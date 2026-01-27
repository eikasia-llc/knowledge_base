import streamlit as st
import os
import sys
from pathlib import Path
import json
import tiktoken

# Add src to path so we can import modules
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from dependency_manager import DependencyManager

# Helper for token counting
@st.cache_resource
def get_encoding():
    return tiktoken.get_encoding("cl100k_base")

# Page Config
st.set_page_config(
    page_title="Knowledge Base Injector",
    page_icon="🧠",
    layout="wide"
)

# Initialize Manager
@st.cache_resource
def get_manager():
    # Root is the parent of src/
    repo_root = current_dir.parent
    return DependencyManager(project_root=repo_root)

manager = get_manager()

# Sidebar
with st.sidebar:
    st.header("Actions")
    if st.button("Sync & Scan Repo"):
        with st.spinner("Scanning repository..."):
            scanned = manager.scan_project()
            manager.update_registry(scanned)
        st.success(f"Scanned {len(scanned)} files.")
    
    st.divider()
    
    # st.subheader("Filter Settings")
    # No filters for now

# Main Content
st.title("🧠 Knowledge Base Injector")
st.markdown("\n\nSelect the knowledge base to get the prompt.\n\n")

# Layout: Output at the top (placeholder)
output_container = st.container()

# Load Registry
try:
    with open(manager.REGISTRY_FILE, 'r') as f:
        registry = json.load(f)
except FileNotFoundError:
    registry = {"files": {}}
    st.warning("Registry not found. Please click 'Sync & Scan Repo'.")

files = registry.get("files", {})

# Categorize Files
categories = {
    "Core": [],
    "Skills": [],
    "Guidelines": [],
    "Protocols": [],
    "Logs": [],
    "Plans": [],
    "Context": [],
    "Uncategorized": []
}

# Core files pattern heuristics
core_filenames = ["README.md", "dependency_registry.json"]

for path, info in files.items():
    lower_path = path.lower()
    
    # Core categorization
    # If it's in the core directory or one of the root core files
    if "content/core" in path or any(path.endswith(c) for c in core_filenames):
        categories["Core"].append(path)
        continue

    # Determine category based on path or metadata (for now simple path heuristics)
    if "agent" in lower_path and "skill" in lower_path:
        categories["Skills"].append(path)
    elif "guideline" in lower_path or "convention" in lower_path:
         categories["Guidelines"].append(path)
    elif "protocol" in lower_path:
         categories["Protocols"].append(path)
    elif "log" in lower_path:
         categories["Logs"].append(path)
    elif "plan" in lower_path:
         categories["Plans"].append(path)
    else:
        # Check if it looks like an Agent definition
        if path.endswith("_AGENT.md"):
             categories["Skills"].append(path)
        else:
             categories["Uncategorized"].append(path)

# UI for Selection
selected_files = []

for category, items in categories.items():
    if items:
        with st.expander(f"{category} ({len(items)})", expanded=(category in ["Core", "Skills", "Protocols"])):
            for item in sorted(items):
                if st.checkbox(item, key=item):
                    selected_files.append(item)


# Output Generation (Rendered into the top container)
if selected_files:
    with output_container:
        st.divider()
        st.markdown("## Generated Prompt Context")
        
        st.markdown("### Output Format")
        output_format = st.radio("Output Format", ["Instruction", "XML", "List"], horizontal=True, label_visibility="collapsed")
        
        # Resolve all dependencies for all selected files
        final_set = []
        seen = set()
        seen_filenames = set()
        
        all_resolved_paths = []
        
        for f in selected_files:
            resolved = manager.resolve_dependencies(f)
            for r in resolved:
                filename = os.path.basename(r)
                if r not in seen and filename not in seen_filenames:
                    seen.add(r)
                    seen_filenames.add(filename)
                    all_resolved_paths.append(r)
        
        # Now generate the text based on the consolidated list
        if output_format == "XML":
            generated_content = "<required_readings>\n"
            for i, f in enumerate(all_resolved_paths, 1):
                 generated_content += f'  <dependency order="{i}">{f}</dependency>\n'
            generated_content += "</required_readings>"
        elif output_format == "List":
            generated_content = "\n".join(f"- {f}" for f in all_resolved_paths)
        else: # Instruction
            generated_content = "The following files are required dependencies. They may be distributed across the codebase, so please **search for these specific filenames** to locate and read them:\n\n"
            for i, f in enumerate(all_resolved_paths, 1):
                filename = os.path.basename(f)
                generated_content += f"{i}. Read {filename}\n"
        
        # State Management: Update session state if filters/selection changed
        # We construct a unique key for the current selection state
        selection_key = f"{sorted(selected_files)}_{output_format}"
        
        if "last_selection_key" not in st.session_state or st.session_state["last_selection_key"] != selection_key:
            st.session_state["prompt_content"] = generated_content
            st.session_state["last_selection_key"] = selection_key
            
        # Layout: Text Area + Metrics
        c1, c2 = st.columns([3, 1])
        
        with c1:
            st.markdown("### Prompt Content")
            
            # Mode Toggler
            mode = st.radio("Mode", ["Copy", "Editing"], horizontal=True, label_visibility="collapsed")
            
            if mode == "Editing":
                user_content = st.text_area("Prompt Content", value=st.session_state["prompt_content"], height=600, label_visibility="collapsed")
                # Update state on every keystroke/change effectively (on rerun)
                st.session_state["prompt_content"] = user_content
            else:
                # Copy Mode
                st.code(st.session_state["prompt_content"], language="text")
                user_content = st.session_state["prompt_content"]
        
        # Token Counting (Using user_content to reflect edits)
        enc = get_encoding()
        prompt_tokens = len(enc.encode(user_content))
        
        file_tokens = 0
        file_token_details = []
        
        for f in all_resolved_paths:
            # Construct absolute path using manager.project_root
            abs_path = manager.project_root / f
            try:
                with open(abs_path, 'r', encoding='utf-8') as file_obj:
                    file_text = file_obj.read()
                    count = len(enc.encode(file_text))
                    file_tokens += count
                    file_token_details.append(f"{os.path.basename(f)}: {count}")
            except Exception:
                file_token_details.append(f"{os.path.basename(f)}: Error reading")

        with c2:
            st.markdown("### Token Stats")
            st.metric("Prompt Instructions", f"~{prompt_tokens}")
            st.metric("Referenced Context", f"~{file_tokens}", help="Estimated tokens of the content within the referenced MD files.")
            
            st.caption("Perdon Fran todavia hay que resolver el tema del metadata overhead 😅")
            
            with st.expander("Details"):
                st.write("\n".join(f"- {d}" for d in file_token_details))

        st.info(f"Selected {len(selected_files)} files resolved to {len(all_resolved_paths)} required readings (including hidden dependencies).")
else:
    with output_container:
        st.info("Select files from the categories below to build your prompt.")
