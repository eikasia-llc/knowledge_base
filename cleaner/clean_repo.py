import sys
import os
import shutil
import subprocess
import tempfile
import argparse

# Add language directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
language_dir = os.path.join(grandparent_dir, 'language')
sys.path.append(language_dir)

try:
    import migrate
except ImportError as e:
    print(f"Error importing language tools: {e}")
    sys.exit(1)

def transform_github_url_to_folder_name(url):
    # Handle branch URLs: https://github.com/user/repo/tree/branch
    if '/tree/' in url:
        base_url = url.split('/tree/')[0]
        name = base_url.rstrip('/').split('/')[-1]
    else:
        name = url.rstrip('/').split('/')[-1]
    
    if name.endswith('.git'):
        name = name[:-4]
    return name

def clean_repo(url):
    # Determine if it's a branch URL
    branch = None
    if '/tree/' in url:
        parts = url.split('/tree/')
        repo_url = parts[0]
        branch = parts[1]
    else:
        repo_url = url
    
    repo_name = transform_github_url_to_folder_name(url)
    print(f"Processing Repository: {repo_name} ({repo_url})")
    if branch:
        print(f"Target Branch: {branch}")

    destination_dir = os.path.join(current_dir, "temprepo_cleaning")
    
    if os.path.exists(destination_dir):
        print(f"Cleaning existing directory: {destination_dir}")
        shutil.rmtree(destination_dir)
        
    os.makedirs(destination_dir)
    print(f"Created directory: {destination_dir}")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning to temporary directory: {temp_dir}")
        try:
            cmd = ['git', 'clone', repo_url, temp_dir]
            if branch:
                cmd.extend(['-b', branch])
                
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"Error: Failed to clone {repo_url} (branch: {branch if branch else 'default'})")
            return

        md_count = 0
        for root, dirs, files in os.walk(temp_dir):
            if '.git' in dirs:
                dirs.remove('.git') # Don't traverse .git
            
            for file in files:
                if file.endswith('.md'):
                    source_path = os.path.join(root, file)
                    target_path = os.path.join(destination_dir, file)
                    
                    # Run migration IN PLACE on the temp file first
                    # If it needed changes, migrate_file returns True
                    try:
                        was_changed = migrate.migrate_file(source_path)
                        
                        if was_changed:
                            print(f"Migrated {file} (updated in place)")
                        else:
                            print(f"Migrated {file} (no changes needed)")
                            
                        if os.path.exists(target_path):
                            print(f"Warning: Overwriting existing file {file} in cleaning directory.")

                        shutil.copy2(source_path, target_path)
                        md_count += 1
                    except Exception as e:
                        print(f"Error checking/migrating {file}: {e}")

        print(f"Successfully cleaned and migrated {md_count} markdown files to {destination_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone repo and migrate markdown files to manager/temprepo_cleaning/")
    parser.add_argument("repo", help="GitHub Repository URL")
    
    # Note: clean_repo does not support -i/-o as it is a specific repo cleaner.
    # It just needs -h (which argparse provides by default).
    
    args = parser.parse_args()
    
    clean_repo(args.repo)
