import os
import json

root_dir = os.getcwd()
registry_path = os.path.join(root_dir, 'dependency_registry.json')

# Map of (file_to_delete) -> (file_to_keep)
DUPLICATES = {
    'content/guidelines/GCCLOUD_GUIDELINE.md': 'content/guidelines/GCLOUD_GUIDELINE.md',
    'content/plans/UNIFIED_NEXUS_ARCHITECTURE_PLAN.md': 'content/plans/KG_UNIFIED_NEXUS_ARCHITECTURE_PLAN.md'
}

def get_markdown_files():
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        if '/.' in root or '/venv' in root or '/node_modules' in root:
            continue
        if '/manager/cleaner' in root:
             continue
        for filename in filenames:
            if filename.endswith('.md'):
                files.append(os.path.join(root, filename))
    return files

def update_links_in_file(filepath, replacements):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for old_rel, new_rel in replacements.items():
            # old_rel is like 'content/guidelines/GCCLOUD_GUIDELINE.md'
            old_base = os.path.basename(old_rel)
            new_base = os.path.basename(new_rel)
            
            # Replace exact paths from root
            new_content = new_content.replace(old_rel, new_rel)
            
            # Replace partial paths (e.g. guidelines/GCCLOUD_GUIDELINE.md)
            old_parts = old_rel.split('/')
            new_parts = new_rel.split('/')
            if len(old_parts) >= 2:
                old_partial = "/".join(old_parts[-2:])
                new_partial = "/".join(new_parts[-2:])
                new_content = new_content.replace(old_partial, new_partial)
                
            # Replace basenames
            new_content = new_content.replace(old_base, new_base)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated links in {os.path.relpath(filepath, root_dir)}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def update_registry(replacements):
    if not os.path.exists(registry_path):
        return
        
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    new_files = {}
    
    def map_path(p):
        if not p: return p
        for old_rel, new_rel in replacements.items():
            if old_rel in p:
                return p.replace(old_rel, new_rel)
            old_base = os.path.basename(old_rel)
            new_base = os.path.basename(new_rel)
            if old_base in p:
                return p.replace(old_base, new_base)
        return p

    for file_key, data in registry.get('files', {}).items():
        # if the node itself is the deleted file, we skip adding it, or rather we probably just remap its key?
        # Actually it's better to NOT keep the deleted file in the registry.
        if file_key in replacements:
            continue # Drop the deleted file entirely from registry
            
        new_key = map_path(file_key)
        new_data = dict(data)
        new_data['path'] = map_path(data.get('path', ''))
        
        new_deps = {}
        for alias, dep_path in new_data.get('dependencies', {}).items():
            new_deps[alias] = map_path(dep_path)
            
        new_data['dependencies'] = new_deps
        new_files[new_key] = new_data
        
    registry['files'] = new_files
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    print("Updated dependency_registry.json")

def execute():
    md_files = get_markdown_files()
    
    for f in md_files:
        update_links_in_file(f, DUPLICATES)
        
    update_registry(DUPLICATES)
    
    for old_file in DUPLICATES.keys():
        abs_old_file = os.path.join(root_dir, old_file)
        if os.path.exists(abs_old_file):
            os.remove(abs_old_file)
            print(f"Deleted duplicate file: {old_file}")

if __name__ == '__main__':
    execute()
