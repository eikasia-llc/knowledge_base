import streamlit as st
import os
import sys
from pathlib import Path
import json

# Add src to path so we can import modules
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from dependency_manager import DependencyManager

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
    
    st.subheader("Filter Settings")
    show_core = st.checkbox("Show Core Dependencies", value=False, help="Show infrastructure files like AGENTS.md or MD_CONVENTIONS.md")

# Main Content
st.title("🧠 Knowledge Base Injector")
st.markdown("Select context modules to inject into your prompt.")

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
    "Skills": [],
    "Guidelines": [],
    "Protocols": [],
    "Logs": [],
    "Plans": [],
    "Context": [],
    "Uncategorized": []
}

core_files = ["AGENTS.md", "MD_CONVENTIONS.md", "README.md", "dependency_registry.json"]

for path, info in files.items():
    # Skip core files if hidden
    is_core = any(path.endswith(c) for c in core_files)
    if is_core and not show_core:
        continue
        
    # Determine category based on path or metadata (for now simple path heuristics)
    # Ideally we should parse 'type' from metadata, but DependencyManager currently extracts only dependencies.
    # We will assume some naming conventions or folder structure for now, 
    # but eventually we should update DependencyManager to return all metadata.
    
    # Simple heuristic fallback
    lower_path = path.lower()
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
        with st.expander(f"{category} ({len(items)})", expanded=(category in ["Skills", "Protocols"])):
            for item in sorted(items):
                if st.checkbox(item, key=item):
                    selected_files.append(item)

# Output Generation
if selected_files:
    st.divider()
    st.subheader("Generated Prompt Context")
    
    output_format = st.radio("Output Format", ["Instruction", "XML", "List"], horizontal=True)
    
    # Resolve all dependencies for all selected files
    final_set = []
    seen = set()
    
    # We iterate selected files and resolve dependencies for each
    # For a robust solution, we should probably merge the lists while preserving order
    # But usually, just concatenating the resolution of each is okay, or we can use a smarter merge.
    # The DependencyManager resolves for a *single* file.
    # Let's resolve each and accumulate unique ones in order.
    
    all_resolved_paths = []
    
    for f in selected_files:
        resolved = manager.resolve_dependencies(f)
        for r in resolved:
            if r not in seen:
                seen.add(r)
                all_resolved_paths.append(r)
    
    # Now generate the text based on the consolidated list
    if output_format == "XML":
        content = "<required_readings>\n"
        for i, f in enumerate(all_resolved_paths, 1):
             content += f'  <dependency order="{i}">{f}</dependency>\n'
        content += "</required_readings>"
    elif output_format == "List":
        content = "\n".join(f"- {f}" for f in all_resolved_paths)
    else: # Instruction
        content = "You must read the following files in order to build your context. You may find them in different folders, but you can do a grep search through the project:\n\n"
        for i, f in enumerate(all_resolved_paths, 1):
            filename = os.path.basename(f)
            content += f"{i}. Read {filename}\n"
            
    st.code(content, language="xml" if output_format == "XML" else "text")
    
    st.info(f"Selected {len(selected_files)} files resolved to {len(all_resolved_paths)} required readings (including hidden dependencies).")

else:
    st.info("Select files from the categories above to build your prompt.")
