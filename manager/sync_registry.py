import os
import json

REGISTRY_PATH = 'dependency_registry.json'

def get_markdown_files(root_dir):
    files = set()
    for root, dirs, filenames in os.walk(root_dir):
        if '/.' in root or '/venv' in root or '/node_modules' in root:
            continue
        if '/manager/cleaner' in root:
            continue
        for filename in filenames:
            if filename.endswith('.md'):
                rel_path = os.path.relpath(os.path.join(root, filename), root_dir)
                files.add(rel_path)
    return files

def sync_registry():
    root_dir = os.getcwd()
    md_files = get_markdown_files(root_dir)
    
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {"files": {}}
        
    registered_files = set(registry.get('files', {}).keys())
    
    missing_from_disk = registered_files - md_files
    missing_from_registry = md_files - registered_files
    
    # 1. Remove stale entries
    for stale in missing_from_disk:
        del registry['files'][stale]
        print(f"Removed stale entry: {stale}")
        
    # 2. Add untracked entries (basic stub)
    for untracked in missing_from_registry:
        registry['files'][untracked] = {
            "path": untracked,
            "dependencies": {}
        }
        print(f"Added untracked file: {untracked}")
        
    # 3. Clean broken markdown dependencies
    # Only remove broken dependencies if they point to .md files that don't exist
    # If they point to .py or .json files, leave them be.
    for path, data in registry['files'].items():
        deps_to_keep = {}
        for alias, dep_path in data.get('dependencies', {}).items():
            if dep_path.endswith('.md'):
                if dep_path in md_files:
                    deps_to_keep[alias] = dep_path
                else:
                    print(f"Removed broken MD dependency in {path}: {alias} -> {dep_path}")
            else:
                deps_to_keep[alias] = dep_path
        data['dependencies'] = deps_to_keep
        
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    print("Registry sync complete.")

if __name__ == '__main__':
    sync_registry()
