import os
import re

def standardize_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    def replacer(match):
        prefix = match.group(1)
        inner = match.group(2)
        # Remove single and double quotes
        inner_clean = inner.replace("'", "").replace('"', "")
        # Standardize spacing around commas just in case
        parts = [p.strip() for p in inner_clean.split(',') if p.strip()]
        return f"{prefix}label: [{', '.join(parts)}]"
        
    # Match `- label: [...]` or `label: [...]`
    new_content, count = re.subn(r'(-?\s*)label:\s*\[(.*?)\]', replacer, content)
    
    if count > 0 and content != new_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

if __name__ == '__main__':
    modified_count = 0
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'temprepo_cleaning' in root or 'repositories' in root or '.gemini' in root or '.pytest_cache' in root:
            continue
            
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                if standardize_labels(file_path):
                    modified_count += 1
                    
    print(f"Standardization complete. Modified {modified_count} files.")
