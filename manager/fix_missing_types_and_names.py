import os
import re
import json
from pathlib import Path

REGISTRY_PATH = 'dependency_registry.json'

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

def add_missing_metadata_and_get_type(file_path):
    filename = os.path.basename(file_path)
    if filename == 'INFRASTRUCTURE.md':
        target_type = 'documentation'
    elif 'ARDUINO' in filename and 'EXPLANATION' not in filename:
        target_type = 'agent_skill'
    elif 'EXPLANATION' in filename:
        target_type = 'guideline'
    else:
        target_type = 'documentation'
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return target_type
        
    has_type = False
    header_idx = -1
    for i, line in enumerate(lines[:30]):
        if line.startswith('# ') and header_idx == -1:
            header_idx = i
        if line.strip().startswith('- type:'):
            has_type = True
            break
            
    changed = False
    
    # 1. Ensure type is set to guideline if EXPLANATION is in the name
    if 'EXPLANATION' in filename and has_type:
        for i, line in enumerate(lines[:30]):
             if line.strip().startswith('- type:'):
                  if 'guideline' not in line:
                      lines[i] = '- type: guideline\n'
                      changed = True
                  target_type = 'guideline'
                  break
        has_type = True

    # 2. Inject missing metadata if it completely lacks a type
    if not has_type and header_idx != -1:
        has_status = any(l.strip().startswith('- status:') for l in lines[:30])
        metadata_to_inject = []
        if not has_status:
             metadata_to_inject.append('- status: active\n')
        metadata_to_inject.append(f'- type: {target_type}\n')
        lines = lines[:header_idx+1] + metadata_to_inject + lines[header_idx+1:]
        changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
    if has_type and not changed:
        # read existing type
        for line in lines[:30]:
            if line.strip().startswith('- type:'):
                 return line.split(':', 1)[1].strip().strip("'").strip('"')
                 
    return target_type

def determine_renames(files):
    renames = {} # old_abs_path -> new_abs_path
    
    for f in files:
        file_type = add_missing_metadata_and_get_type(f)
        
        filename = os.path.basename(f)
        if filename in DO_NOT_RENAME:
            continue
            
        name_without_ext = filename[:-3]
        
        # Rule: replace EXPLANATION with GUIDELINE
        if 'EXPLANATION' in name_without_ext:
            name_without_ext = name_without_ext.replace('EXPLANATION', 'GUIDELINE')
            
        # Rule: replace - with _
        if '-' in name_without_ext:
            name_without_ext = name_without_ext.replace('-', '_')
            
        expected_suffix = TYPE_SUFFIX_MAP.get(file_type)
        if expected_suffix:
            changes_made = True
            while changes_made:
                changes_made = False
                for s in TYPE_SUFFIX_MAP.values():
                    if name_without_ext.endswith(s):
                        name_without_ext = name_without_ext[:-len(s)]
                        changes_made = True
                if name_without_ext.endswith('_AGENT'):
                    name_without_ext = name_without_ext[:-len('_AGENT')]
                    changes_made = True
                    
            name_without_ext += expected_suffix
            
        new_name = name_without_ext + ".md"

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
        if old_base in new_content:
            pattern = re.compile(r'\b' + re.escape(old_base) + r'\b')
            if pattern.search(new_content):
                new_content = pattern.sub(new_base, new_content)
                has_changes = True
            
            old_no_ext = old_base[:-3]
            new_no_ext = new_base[:-3]
            pattern2 = re.compile(r'\b' + re.escape(old_no_ext) + r'\b')
            if pattern2.search(new_content):
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
    
    def map_path(p):
        for old_p, new_p in rename_map_rel.items():
            if old_p in p:
                return p.replace(old_p, new_p)
            elif os.path.basename(old_p) in p:
                return p.replace(os.path.basename(old_p), os.path.basename(new_p))
        return p

    for file_key, data in registry.get('files', {}).items():
        new_key = map_path(file_key)
        new_data = dict(data)
        new_data['path'] = map_path(data.get('path', ''))
        
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
    
    renames = determine_renames(files)
    
    if not renames:
        print("No files need renaming.")
        return
        
    print(f"Plan to rename {len(renames)} files. Dry Run: {dry_run}")
    
    rename_map_basenames = {os.path.basename(k): os.path.basename(v) for k, v in renames.items()}
    rename_map_rel = {os.path.relpath(k, root): os.path.relpath(v, root) for k, v in renames.items()}
    
    for old_path, new_path in renames.items():
        old_no_ext = os.path.basename(old_path)[:-3]
        new_no_ext = os.path.basename(new_path)[:-3]
        update_file_headers(old_path, old_no_ext, new_no_ext, dry_run)
        
    for f in files:
        update_links(f, rename_map_basenames, dry_run)
        
    update_registry(root, rename_map_rel, dry_run)
    
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
