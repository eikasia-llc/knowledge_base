import os
import re
from pathlib import Path

# --- Configuration ---
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

METADATA_START = re.compile(r'^(-\s+\w+:\s+.*|-\s+id:\s+.*)$')

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

def analyze():
    root = os.getcwd()
    files = get_files_to_process(root)
    
    proposed_renames = []
    
    for f in files:
        filename = os.path.basename(f)
        if filename in DO_NOT_RENAME:
            continue
            
        file_type = extract_type(f)
        if not file_type:
            continue
            
        expected_suffix = TYPE_SUFFIX_MAP.get(file_type)
        if not expected_suffix:
            # We don't map this type
            continue
            
        name_without_ext = filename[:-3]
        
        if file_type == 'agent_skill':
            # Replace _AGENT with _SKILL, or append
            if name_without_ext.endswith('_AGENT'):
                new_name = name_without_ext[:-6] + '_SKILL.md'
            elif not name_without_ext.endswith('_SKILL'):
                new_name = name_without_ext + '_SKILL.md'
            else:
                new_name = filename
                
        elif file_type == 'log':
            if not name_without_ext.endswith('_LOG') and not name_without_ext.endswith('_LOGS'):
                new_name = name_without_ext + '_LOG.md'
            else:
                new_name = filename
                
        elif file_type == 'plan':
            if not name_without_ext.endswith('_PLAN'):
                new_name = name_without_ext + '_PLAN.md'
            else:
                new_name = filename
                
        elif file_type == 'guideline':
             if not name_without_ext.endswith('_GUIDE') and not name_without_ext.endswith('_GUIDELINE'):
                 new_name = name_without_ext + '_GUIDELINE.md'
             else:
                 new_name = filename
                 
        elif file_type == 'documentation':
             if not name_without_ext.endswith('_DOC') and not name_without_ext.endswith('TION'):
                 new_name = name_without_ext + '_DOC.md'
             else:
                 new_name = filename
        else:
             new_name = filename

        if new_name != filename:
            proposed_renames.append((f, new_name, file_type))

    print(f"Total files capable of renaming: {len(proposed_renames)}")
    for f, new_name, ftype in proposed_renames:
        print(f"[{ftype}] {os.path.basename(f)} -> {new_name}")

if __name__ == '__main__':
    analyze()
