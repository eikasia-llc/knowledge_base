
import os
import re
import sys
from pathlib import Path

# --- Configuration ---
ALLOWED_TYPES = {
    'agent_skill', 
    'log', 
    'guideline', 
    'plan', 
    'task', 
    'documentation'
}

MANDATORY_FIELDS = {'status', 'type'}

TYPE_MAPPING = {
    'protocol': 'guideline',
    'context': 'documentation',
    'recurring': 'plan',
    # 'task' is preserved
}

# --- Regex Patterns ---
HEADER_PATTERN = re.compile(r'^(#+)\s+(.+)$')
METADATA_START = re.compile(r'^(-\s+\w+:\s+.*|-\s+id:\s+.*)$')
CONTENT_SEPARATOR = "<!-- content -->"

def get_files_to_process(root_dir):
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        # Exclude hidden directories and venv
        if '/.' in root or '/venv' in root or '/node_modules' in root:
            continue
        # Exclude cleaner repositories
        if '/manager/cleaner/repositories' in root or '/manager/cleaner/temprepo' in root:
            continue
            
        for filename in filenames:
            if filename.endswith('.md'):
                files.append(os.path.join(root, filename))
    return files

def infer_label(file_path, existing_type):
    labels = set()
    path = Path(file_path)
    
    # Path-based inference
    if 'agents' in path.parts:
        labels.add('agent')
    if 'plans' in path.parts:
        labels.add('planning')
    if 'logs' in path.parts or 'LOG' in path.name:
        labels.add('log')
        
    # Type-based inference
    if existing_type == 'recurring':
        labels.add('recurring')
    if existing_type == 'protocol':
        labels.add('protocol')
        
    return list(labels)

def infer_type(file_path):
    path = Path(file_path)
    if 'agents' in path.parts or 'AGENT' in path.name:
        return 'agent_skill'
    if 'plans' in path.parts:
        return 'plan'
    if 'guidelines' in path.parts:
        return 'guideline'
    if 'logs' in path.parts or 'LOG' in path.name:
        return 'log'
    return 'documentation'

def process_file(file_path, dry_run=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_metadata = False
    metadata_buffer = []
    has_changes = False
    
    # Simple state machine to find headers and their metadata blocks
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Detect Header
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            new_lines.append(line)
            i += 1
            
            # 2. Look ahead for metadata
            metadata_block = []
            while i < len(lines):
                next_line = lines[i]
                stripped = next_line.strip()
                
                # Check for metadata line
                if (stripped.startswith('- ') and ':' in stripped) or (stripped.startswith('id:')):
                     metadata_block.append(next_line)
                     i += 1
                # Check for separator or empty lines (legacy) acting as separator
                elif stripped == CONTENT_SEPARATOR or stripped == '':
                    # Determine if this was actually a metadata block
                     if metadata_block:
                         break # End of metadata
                     else:
                        # No metadata found immediately after header
                        new_lines.append(next_line)
                        i += 1
                        break
                else:
                    # Non-metadata line encountered
                    break
            
            # 3. Process Metadata if found
            if metadata_block:
                processed_metadata, changed = process_metadata(metadata_block, file_path)
                
                if changed:
                    has_changes = True
                    if dry_run:
                        print(f"--- {file_path} ---")
                        print("Original:")
                        print("".join(metadata_block).strip())
                        print("Proposed:")
                        print("".join(processed_metadata).strip())
                        print("----------------")

                new_lines.extend(processed_metadata)
                    
                # Ensure separator exists
                if i < len(lines) and lines[i].strip() == CONTENT_SEPARATOR:
                     new_lines.append(lines[i]) # Keep existing separator
                     i += 1
                else:
                    if changed: # Only add separator if we touched metadata and it wasn't there
                         new_lines.append(CONTENT_SEPARATOR + "\n")
                         has_changes = True

            continue
            
        new_lines.append(line)
        i += 1

    if has_changes and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Updated: {file_path}")

def process_metadata(metadata_lines, file_path):
    parsed = {}
    key_order = []
    
    # Parse existing
    for line in metadata_lines:
        parts = line.strip().lstrip('- ').split(':', 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            parsed[key] = value
            key_order.append(key)
            
    changes = False
    
    # 1. Update Type
    original_type = parsed.get('type')
    current_type = original_type
    
    if current_type:
        current_type = current_type.strip('"').strip("'")
        if current_type not in ALLOWED_TYPES:
            new_type = TYPE_MAPPING.get(current_type)
            if new_type:
                parsed['type'] = new_type
                changes = True
            else:
                 # Fallback based on file path if verify fails
                 inferred = infer_type(file_path)
                 parsed['type'] = inferred
                 changes = True
    else:
        # Missing type
        inferred = infer_type(file_path)
        parsed['type'] = inferred
        changes = True
        
    # Ensure 'type' is in key_order if it wasn't before
    if 'type' not in key_order:
        # Insert after status if status exists
        if 'status' in key_order:
            idx = key_order.index('status')
            key_order.insert(idx + 1, 'type')
        else:
            key_order.insert(0, 'type')

    # 2. Update Label
    existing_labels = parsed.get('label', '[]')
    # Basic list parsing (removed brackets)
    current_labels_list = [l.strip().strip("'").strip('"') for l in existing_labels.strip("[]").split(',') if l.strip()]
    
    inferred = infer_label(file_path, original_type)
    
    # Add inferred labels if not present
    updated_labels_set = set(current_labels_list)
    for l in inferred:
        if l not in updated_labels_set:
            updated_labels_set.add(l)
            changes = True
            
    if updated_labels_set:
        # Sort for determinism
        final_list = sorted(list(updated_labels_set))
        # Format as string list
        parsed['label'] = "[" + ", ".join([f"'{l}'" for l in final_list]) + "]"
        if 'label' not in key_order:
            key_order.append('label')
    
    # Reconstruct
    new_lines = []
    for key in key_order:
        val = parsed[key]
        if key == 'status' and 'active' not in val and 'done' not in val:
             # Just a check, implementation is to preserve status unless invalid? 
             # We only touch type and label per instructions.
             pass
             
        new_lines.append(f"- {key}: {val}\n")
        
    return new_lines, changes

if __name__ == "__main__":
    dry_run = '--dry-run' in sys.argv
    root = os.getcwd()
    
    if dry_run:
        print("Running in DRY RUN mode. No files will be changed.")
        
    # print(f"Scanning {root}...")
    files = get_files_to_process(root)
    # print(f"Found {len(files)} markdown files.")
    
    for f in files:
        # Skip this script's own convention file logic updates to avoid loops if re-run
        if 'MD_CONVENTIONS.md' in f:
             continue
        process_file(f, dry_run=dry_run)
