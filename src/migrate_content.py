import os
import shutil
from pathlib import Path
import json

# Setup paths
script_dir = Path(__file__).parent.resolve()
kb_repo = script_dir.parent # knowldege_bases_repo
# central_planner/manager/knowledge_bases/knowldege_bases_repo/src -> ... -> central_planner
repo_root = kb_repo.parent.parent.parent 
content_dir = kb_repo / "content"

# Ensure content dirs exist
categories = {
    "agents": content_dir / "agents",
    "plans": content_dir / "plans",
    "guidelines": content_dir / "guidelines",
    "root": content_dir / "root",
    "logs": content_dir / "logs",
    "misc": content_dir / "misc"
}

for p in categories.values():
    p.mkdir(parents=True, exist_ok=True)

# Define mapping rules (glob -> category)
# Order matters!
rules = [
    ("**/AGENTS.md", "guidelines"),
    ("**/MD_CONVENTIONS.md", "guidelines"),
    ("**/README.md", "root"),
    
    # Specific Agents
    ("**/MANAGER_AGENT.md", "agents"),
    ("**/CLEANER_AGENT.md", "agents"),
    ("AI_AGENTS/**/*.md", "agents"),
    
    # Plans
    ("**/MASTER_PLAN.md", "plans"),
    ("**/subplans/*.md", "plans"),
    ("language/example/*.md", "plans"),
    
    # Logs
    ("**/*LOG*.md", "logs"),
    ("**/MEETING_NOTES.md", "logs"),
    
    # Catch-all
    ("**/*.md", "misc")
]

# Track processed files to avoid duplicates (if multiple globs match)
processed = set()
file_map = {} # original_abs_path -> new_abs_path

def get_category_path(file_path):
    # Try to match rules
    # file_path is absolute or relative to repo_root? 
    # Let's ensure we match against relative path for globbing
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        rel_path = file_path 
    
    for pattern, cat_key in rules:
        if rel_path.match(pattern):
             return categories[cat_key]
    return categories["misc"]

print(f"Scanning {repo_root}...")

# PASS 1: Calculate destinations
found_files = []
for root, dirs, files in os.walk(repo_root):
    if "knowldege_bases_repo" in root:
        continue
    
    for file in files:
        if not file.endswith(".md"):
            continue
            
        src_path = (Path(root) / file).resolve()
        
        # Skip hidden files/dirs
        if any(p.startswith(".") for p in src_path.parts):
            continue
            
        if src_path in processed:
            continue
            
        category_dir = get_category_path(src_path)
        dest_path = category_dir / file
        
        # Handle duplicate filenames
        if dest_path in file_map.values(): # naive check, might need better collision handling
             parent_name = src_path.parent.name
             dest_path = category_dir / f"{parent_name}_{file}"
        
        file_map[src_path] = dest_path
        found_files.append(src_path)
        processed.add(src_path)

print(f"Found {len(found_files)} files. Starting migration and dependency rewriting...")

# PASS 2: Copy and Rewrite
from md_parser import MarkdownParser
parser_obj = MarkdownParser()

for src_path in found_files:
    dest_path = file_map[src_path]
    
    # Read and parse
    try:
        root_node = parser_obj.parse_file(src_path)
        
        # rewritten deps
        new_deps = {}
        original_deps = root_node.metadata.get("context_dependencies", {})
        
        if isinstance(original_deps, dict):
            for alias, dep_rel_path in original_deps.items():
                # Resolve original absolute path of dependency
                # dep_rel_path is relative to src_path
                try:
                    dep_abs_path = (src_path.parent / dep_rel_path).resolve()
                except Exception:
                    print(f"  Warning: Could not resolve dep {dep_rel_path} in {src_path.name}")
                    continue
                
                # Find where this dependency moved to
                if dep_abs_path in file_map:
                    new_dep_dest = file_map[dep_abs_path]
                    # Calculate new relative path from dest_path to new_dep_dest
                    # dest_path is e.g. .../content/agents/AGENT.md
                    # new_dep_dest is e.g. .../content/guidelines/CONVENTIONS.md
                    try:
                        new_rel_path = os.path.relpath(new_dep_dest, dest_path.parent)
                        new_deps[alias] = new_rel_path
                    except ValueError:
                         new_deps[alias] = str(new_dep_dest)
                else:
                    print(f"  Warning: Dependency {dep_rel_path} not found in migration map for {src_path.name}")
                    # Keep original? Or drop? Let's keep original just in case
                    new_deps[alias] = dep_rel_path
            
            # Update metadata
            root_node.metadata["context_dependencies"] = new_deps
            
        # Write to destination
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(root_node.to_markdown())
            
        print(f"Migrated {src_path.name} -> {dest_path.parent.name}/{dest_path.name}")

    except Exception as e:
        print(f"Failed to migrate {src_path.name}: {e}")
        # Fallback copy
        shutil.copy2(src_path, dest_path)

print("Migration complete. Run 'dependency_manager.py scan' next.")
