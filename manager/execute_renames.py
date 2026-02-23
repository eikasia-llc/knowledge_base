import os
import re
import json
from pathlib import Path

# --- Configuration (Copied from analysis) ---
TYPE_SUFFIX_MAP = {
    'agent_skill': '_SKILL',
    'log': '_LOG',
    'guideline': '_GUIDELINE',
    'plan': '_PLAN',
    'task': '_TASK',
    'documentation': '_DOC'
}

DO_NOT_RENAME = {
    'README.md',
    'MD_CONVENTIONS.md',
    'AGENTS.md',
    'HOUSEKEEPING.md'
}

REGISTRY_PATH = 'dependency_registry.json'

def get_files_to_process(root_dir):
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        if '/.' in root or '/venv' in root or '/node_modules' in root:
            continue
        if '/manager/cleaner/repositories' in root or '/manager/cleaner/temprepo' in root:
            continue
        for filename in filenames:
            if filename.endswith('.md'):
                files.append(os.path.join(root, filename))
    return files

def extract_type(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if line.startswith('#'):
                for j in range(i+1, min(i+10, len(lines))):
                    stripped = lines[j].strip()
                    if stripped.startswith('- type:'):
                        return stripped.split(':', 1)[1].strip().strip("'").strip('"')
                    elif stripped == '<!-- content -->':
                        break
        return None
    except Exception:
        return None

def determine_renames(files):
    renames = {} # old_abs_path -> new_abs_path
    
    for f in files:
        filename = os.path.basename(f)
        if filename in DO_NOT_RENAME:
            continue
            
        file_type = extract_type(f)
        if not file_type:
            continue
            
        expected_suffix = TYPE_SUFFIX_MAP.get(file_type)
        if not expected_suffix:
            continue
            
        name_without_ext = filename[:-3]
        new_name = filename
        
        if file_type == 'agent_skill':
            if name_without_ext.endswith('_AGENT'):
                new_name = name_without_ext[:-6] + '_SKILL.md'
            elif not name_without_ext.endswith('_SKILL'):
                new_name = name_without_ext + '_SKILL.md'
                
        elif file_type == 'log':
            if not name_without_ext.endswith('_LOG') and not name_without_ext.endswith('_LOGS'):
                new_name = name_without_ext + '_LOG.md'
                
        elif file_type == 'plan':
            if not name_without_ext.endswith('_PLAN'):
                new_name = name_without_ext + '_PLAN.md'
                
        elif file_type == 'guideline':
             if not name_without_ext.endswith('_GUIDE') and not name_without_ext.endswith('_GUIDELINE'):
                 new_name = name_without_ext + '_GUIDELINE.md'
                 
        elif file_type == 'documentation':
             if not name_without_ext.endswith('_DOC') and not name_without_ext.endswith('TION'):
                 new_name = name_without_ext + '_DOC.md'

        if new_name != filename:
            new_path = os.path.join(os.path.dirname(f), new_name)
            renames[f] = new_path
            
    return renames

def update_file_headers(file_path, old_name_no_ext, new_name_no_ext, dry_run=True):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
        
    # Replace # OLD_NAME with # NEW_NAME at the start of the file or after empty lines
    pattern = re.compile(rf'^(#\s+){re.escape(old_name_no_ext)}(\s*)$', re.MULTILINE | re.IGNORECASE)
    
    if pattern.search(content):
        new_content = pattern.sub(rf'\g<1>{new_name_no_ext}\g<2>', content)
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print(f"  [Dry Run] Updated header in {os.path.basename(file_path)}")

def update_links(file_path, rename_map_basenames, dry_run=True):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
        
    new_content = content
    has_changes = False
    
    for old_base, new_base in rename_map_basenames.items():
        # Replace occurrences of old_base (e.g., in links []() or text)
        # We need to be careful not to replace parts of other words
        if old_base in new_content:
            # Replace exactly matching words / basenames
            # e.g., CLEANER_AGENT.md -> CLEANER_SKILL.md
            pattern = re.compile(r'\b' + re.escape(old_base) + r'\b')
            if pattern.search(new_content):
                new_content = pattern.sub(new_base, new_content)
                has_changes = True
            
            # Also try without .md for links or references like "CLEANER_AGENT" -> "CLEANER_SKILL"
            old_no_ext = old_base[:-3]
            new_no_ext = new_base[:-3]
            pattern2 = re.compile(r'\b' + re.escape(old_no_ext) + r'\b')
            if pattern2.search(new_content):
                # Avoid matching if it was already part of the first replacement, but regex handles raw text.
                # Actually, replace backwards to be safe, but a simple replace is usually fine for these specific caps names.
                # Let's just do a naive regex word boundary replace for the base name too.
                # But only if it looks like a reference (e.g. uppercase). 
                # Since these are usually uppercase, it's fairly safe.
                new_content = pattern2.sub(new_no_ext, new_content)
                has_changes = True

    if has_changes and content != new_content:
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print(f"  [Dry Run] Updated links/references in {os.path.basename(file_path)}")

def update_registry(root_dir, rename_map_rel, dry_run=True):
    path = os.path.join(root_dir, REGISTRY_PATH)
    if not os.path.exists(path):
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    new_files = {}
    
    # helper to map paths
    def map_path(p):
        for old_p, new_p in rename_map_rel.items():
            if old_p in p:
                return p.replace(old_p, new_p)
            elif os.path.basename(old_p) in p:
                # E.g. "../agents/CLEANER_AGENT.md"
                return p.replace(os.path.basename(old_p), os.path.basename(new_p))
        return p

    for file_key, data in registry.get('files', {}).items():
        new_key = map_path(file_key)
        new_data = dict(data)
        new_data['path'] = map_path(data.get('path', ''))
        
        # update dependencies
        new_deps = {}
        for alias, dep_path in new_data.get('dependencies', {}).items():
            new_deps[alias] = map_path(dep_path)
            
        new_data['dependencies'] = new_deps
        new_files[new_key] = new_data
        
    registry['files'] = new_files
    
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
    else:
        print(f"  [Dry Run] Updated internal registry paths.")

def execute(dry_run=True):
    root = os.getcwd()
    files = get_files_to_process(root)
    
    # 1. Determine renames
    renames = determine_renames(files)
    
    if not renames:
        print("No files need renaming.")
        return
        
    print(f"Plan to rename {len(renames)} files. Dry Run: {dry_run}")
    
    rename_map_basenames = {os.path.basename(k): os.path.basename(v) for k, v in renames.items()}
    rename_map_rel = {os.path.relpath(k, root): os.path.relpath(v, root) for k, v in renames.items()}
    
    # 2. Update Headers in files BEFORE renaming (easier to read)
    for old_path, new_path in renames.items():
        old_no_ext = os.path.basename(old_path)[:-3]
        new_no_ext = os.path.basename(new_path)[:-3]
        update_file_headers(old_path, old_no_ext, new_no_ext, dry_run)
        
    # 3. Update cross-references in ALL markdown files
    for f in files:
        update_links(f, rename_map_basenames, dry_run)
        
    # 4. Update dependency registry
    update_registry(root, rename_map_rel, dry_run)
    
    # 5. Actually rename the files
    for old_path, new_path in renames.items():
        if not dry_run:
            os.rename(old_path, new_path)
            print(f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
        else:
            print(f"  [Dry Run] Rename: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")

if __name__ == '__main__':
    import sys
    dry_run = '--execute' not in sys.argv
    execute(dry_run)
