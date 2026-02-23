import os
import difflib

def get_files(root_dir):
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

def find_duplicates():
    root = os.getcwd()
    files = get_files(os.path.join(root, 'content'))
    
    names = [os.path.basename(f) for f in files]
    
    print("Potential duplicates based on name similarity:")
    seen = set()
    for i in range(len(names)):
        if i in seen: continue
        matches = difflib.get_close_matches(names[i], names, n=5, cutoff=0.7)
        if len(matches) > 1:
            print(f"Group: {', '.join(matches)}")
            for j in range(len(names)):
                 if names[j] in matches:
                      seen.add(j)

if __name__ == '__main__':
    find_duplicates()
