import os
import re
import json
import shutil
from pathlib import Path

REGISTRY_PATH = 'dependency_registry.json'

TYPE_FOLDER_MAP = {
    'agent_skill': 'content/skills',
    'log': 'content/logs',
    'guideline': 'content/guidelines',
    'plan': 'content/plans',
    'task': 'content/tasks',
    'documentation': 'content/documentation'
}

# Root-level files that should not be moved
DO_NOT_MOVE = {
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

def extract_type(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines[:30]):
            if line.strip().startswith('- type:'):
                return line.split(':', 1)[1].strip().strip("'").strip('"')
    except Exception:
        pass
    return None

def determine_moves(files, root_dir):
    moves = {} # old_abs_path -> new_abs_path
    
    for f in files:
        filename = os.path.basename(f)
        rel_path = os.path.relpath(f, root_dir)
        
        # Don't move specific root files or files in language/example
        if filename in DO_NOT_MOVE or 'manager/language/example' in rel_path or 'manager/cleaner' in rel_path:
             continue
            
        file_type = extract_type(f)
        if not file_type:
            continue
            
        target_folder = TYPE_FOLDER_MAP.get(file_type)
        if not target_folder:
            continue
            
        # Check if already in the right folder
        target_dir_abs = os.path.join(root_dir, target_folder)
        current_dir_abs = os.path.dirname(f)
        
        if current_dir_abs != target_dir_abs:
            new_path = os.path.join(target_dir_abs, filename)
            # handle case where filename exists in target unexpectedly? 
            # for now assume unique names
            moves[f] = new_path
            
    return moves

def update_links(file_path, moves, root_dir, dry_run=True):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
        
    new_content = content
    has_changes = False
    
    # We need to replace occurrences of old relative paths with new ones.
    # Links might be: `../agents/CLEANER_SKILL.md`
    # Or `content/agents/CLEANER_SKILL.md`
    # Or just `CLEANER_SKILL.md` (which we shouldn't change the basename of, just the path if explicit)
    
    # A simpler heuristic for this codebase: replace `content/agents/` with `content/skills/`
    # But specifically, we can just look for the known string replacement.
    
    for old_path, new_path in moves.items():
         old_rel = os.path.relpath(old_path, root_dir)
         new_rel = os.path.relpath(new_path, root_dir)
         
         # 1. Exact relative paths from root: `content/agents/X.md` -> `content/skills/X.md`
         if old_rel in new_content:
             new_content = new_content.replace(old_rel, new_rel)
             has_changes = True
             
         # 2. Relative paths from other dirs: `../agents/X.md` -> `../skills/X.md`
         old_parts = old_rel.split('/')
         new_parts = new_rel.split('/')
         if len(old_parts) >= 2 and len(new_parts) >= 2:
              # e.g., agents/X.md -> skills/X.md
              old_partial = "/".join(old_parts[-2:])
              new_partial = "/".join(new_parts[-2:])
              if old_partial in new_content:
                  new_content = new_content.replace(old_partial, new_partial)
                  has_changes = True

    if has_changes and content != new_content:
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print(f"  [Dry Run] Updated links/references in {os.path.basename(file_path)}")

def update_registry(root_dir, moves, dry_run=True):
    path = os.path.join(root_dir, REGISTRY_PATH)
    if not os.path.exists(path):
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    new_files = {}
    
    move_map_rel = {os.path.relpath(k, root_dir): os.path.relpath(v, root_dir) for k, v in moves.items()}
    
    def map_path(p):
        # same logic as links
        if not p: return p
        if p in move_map_rel:
            return move_map_rel[p]
            
        for old_rel, new_rel in move_map_rel.items():
            if old_rel in p:
                return p.replace(old_rel, new_rel)
            
            old_parts = old_rel.split('/')
            new_parts = new_rel.split('/')
            if len(old_parts) >= 2 and len(new_parts) >= 2:
                 old_partial = "/".join(old_parts[-2:])
                 new_partial = "/".join(new_parts[-2:])
                 if old_partial in p:
                     return p.replace(old_partial, new_partial)
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
    
    moves = determine_moves(files, root)
    if not moves:
        print("No files need to be moved.")
        return
        
    print(f"Plan to move {len(moves)} files. Dry Run: {dry_run}")
    
    # Update links in all files
    for f in files:
        update_links(f, moves, root, dry_run)
        
    update_registry(root, moves, dry_run)
    
    # Actually move
    for old_path, new_path in moves.items():
        if not dry_run:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
            print(f"Moved: {os.path.relpath(old_path, root)} -> {os.path.relpath(new_path, root)}")
        else:
             print(f"  [Dry Run] Move: {os.path.relpath(old_path, root)} -> {os.path.relpath(new_path, root)}")

if __name__ == '__main__':
    import sys
    dry_run = '--execute' not in sys.argv
    execute(dry_run)
