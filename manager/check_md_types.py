import os

ALLOWED_TYPES = {
    'agent_skill', 
    'log', 
    'guideline', 
    'plan', 
    'task', 
    'documentation'
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

def check_file_type(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Look ahead for metadata
                for j in range(i+1, min(i+10, len(lines))):
                    stripped = lines[j].strip()
                    if stripped.startswith('- type:'):
                        ftype = stripped.split(':', 1)[1].strip().strip("'").strip('"')
                        if ftype in ALLOWED_TYPES:
                            return 'valid', ftype
                        else:
                            return 'invalid_type', ftype
                    elif stripped == '<!-- content -->':
                        # metadata block over, no type found
                        return 'missing_type', None
                # If we exhausted 10 lines and didn't find the separator or type
                return 'missing_type', None
                
        # If we loop through the whole file and find no header... well that's an issue too
        return 'no_header', None
    except Exception as e:
        return 'error', str(e)

def scan():
    root = os.getcwd()
    files = get_files_to_process(root)
    
    missing_type = []
    invalid_type = []
    no_header = []
    error = []
    
    for f in files:
        status, detail = check_file_type(f)
        rel_path = os.path.relpath(f, root)
        
        if status == 'missing_type':
            missing_type.append(rel_path)
        elif status == 'invalid_type':
            invalid_type.append((rel_path, detail))
        elif status == 'no_header':
            no_header.append(rel_path)
        elif status == 'error':
            error.append((rel_path, detail))
            
    print("--- SCAN RESULTS ---")
    print(f"Total files scanned: {len(files)}\n")
    
    if invalid_type:
        print(f"[{len(invalid_type)}] Files with INCORRECT/UNCLEAR type:")
        for path, t in invalid_type:
            print(f"  - {path} (type: '{t}')")
            
    if missing_type:
        print(f"\n[{len(missing_type)}] Files MISSING type:")
        for path in missing_type:
            print(f"  - {path}")
            
    if no_header:
        print(f"\n[{len(no_header)}] Files with NO HEADER:")
        for path in no_header:
            print(f"  - {path}")
            
    if error:
        print(f"\n[{len(error)}] Files with ERRORS during parse:")
        for path, err in error:
            print(f"  - {path} ({err})")
            
    if not missing_type and not invalid_type and not no_header and not error:
        print("All analyzed files have a valid type!")

if __name__ == '__main__':
    scan()
