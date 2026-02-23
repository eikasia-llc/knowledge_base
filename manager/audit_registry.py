import os
import json

REGISTRY_PATH = 'dependency_registry.json'

def get_markdown_files(root_dir):
    files = set()
    for root, dirs, filenames in os.walk(root_dir):
        if '/.' in root or '/venv' in root or '/node_modules' in root:
            continue
        if '/manager/cleaner/repositories' in root or '/manager/cleaner/temprepo' in root:
            continue
        for filename in filenames:
            if filename.endswith('.md'):
                rel_path = os.path.relpath(os.path.join(root, filename), root_dir)
                files.add(rel_path)
    return files

def audit_registry():
    root_dir = os.getcwd()
    md_files = get_markdown_files(root_dir)
    
    if not os.path.exists(REGISTRY_PATH):
        print(f"Registry not found: {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    registered_files = set(registry.get('files', {}).keys())
    
    # 1. Missing from disk (in registry but file doesn't exist)
    missing_from_disk = registered_files - md_files
    
    # 2. Missing from registry (file exists but not in registry)
    missing_from_registry = md_files - registered_files
    
    print("\n=== Audit Report ===")
    print(f"Files on disk: {len(md_files)}")
    print(f"Files in registry: {len(registered_files)}")
    
    print(f"\nMissing from disk (stale entries): {len(missing_from_disk)}")
    for f in sorted(missing_from_disk):
        print(f"  - {f}")
        
    print(f"\nMissing from registry (untracked files): {len(missing_from_registry)}")
    for f in sorted(missing_from_registry):
        print(f"  - {f}")

    # 3. Check internal dependencies (dependencies pointing to missing paths)
    broken_deps = []
    files = registry.get('files', {})
    for path, data in files.items():
        if path in missing_from_disk:
            continue
        for alias, dep_path in data.get('dependencies', {}).items():
            if dep_path not in md_files and dep_path not in files:
                broken_deps.append((path, alias, dep_path))
                
    print(f"\nBroken dependencies: {len(broken_deps)}")
    for source, alias, dest in sorted(broken_deps):
        print(f"  - {source} -> ({alias}) {dest}")

if __name__ == '__main__':
    audit_registry()
