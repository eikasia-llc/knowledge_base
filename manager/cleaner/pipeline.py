#!/usr/bin/env python3
import os
import sys
import datetime
import subprocess

# Ensure we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir) # for clean_repo
sys.path.append(os.path.join(parent_dir, 'language')) # for apply_types logic if needed, but we'll use subprocess for isolation

import clean_repo

def main():
    print("Starting Cleaning Pipeline...")
    
    # 1. Read Repo List
    repo_list_path = os.path.join(current_dir, 'toclean_repolist.txt')
    if not os.path.exists(repo_list_path):
        print(f"Error: {repo_list_path} not found.")
        sys.exit(1)
        
    with open(repo_list_path, 'r') as f:
        repos = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
    if not repos:
        print("No repositories found in list.")
        sys.exit(0)
        
    print(f"Found {len(repos)} repositories to process.")
    
    # 2. Process Repos
    total_files = 0
    logs = []
    
    for repo_url in repos:
        print(f"\nProcessing {repo_url}...")
        try:
            # clean_repo returns (count, destination_dir)
            md_count, dest_dir = clean_repo.clean_repo(repo_url)
            total_files += md_count
            
            if md_count > 0:
                # 3. Apply Types to the Staging Directory
                print(f"Running type standardization on {dest_dir}...")
                apply_types_script = os.path.join(parent_dir, 'language', 'apply_types.py')
                
                cmd = [sys.executable, apply_types_script, '--target_dir', dest_dir]
                subprocess.run(cmd, check=True)
                
                log_entry = f"- {datetime.datetime.now().isoformat()}: Imported {md_count} files from {repo_url}"
                logs.append(log_entry)
            else:
                print("No markdown files found or imported.")
                
        except Exception as e:
            print(f"Error processing {repo_url}: {e}")
            logs.append(f"- {datetime.datetime.now().isoformat()}: Failed to import {repo_url}. Error: {e}")

    # 4. Update Logs
    log_file = os.path.join(current_dir, 'CLEANING_LOGS.md')
    print(f"\nUpdating logs at {log_file}...")
    
    try:
        with open(log_file, 'a') as f:
            for entry in logs:
                f.write(entry + "\n")
                
        # Also print to console
        # print("\nPipeline Complete. Summary:")
        # for entry in logs:
        #     print(entry)
            
    except Exception as e:
        print(f"Error writing logs: {e}")

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
