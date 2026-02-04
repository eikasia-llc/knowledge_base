import os
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GitManager:
    def __init__(self, repo_url, target_path, github_token=None):
        self.repo_url = repo_url
        self.target_path = Path(target_path)
        self.github_token = github_token.strip() if github_token else None
        
        # Format URL with token if provided
        if self.github_token and repo_url.startswith("https://"):
            self.authenticated_url = repo_url.replace("https://", f"https://x-access-token:{self.github_token}@")
        else:
            self.authenticated_url = repo_url

    def _run_command(self, command, cwd=None):
        """Helper to run git commands and capture output."""
        try:
            result = subprocess.run(
                command, 
                cwd=cwd or str(self.target_path), 
                check=True, 
                capture_output=True, 
                text=True
            )
            return True, result.stdout + result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout + e.stderr

    def initialize_repo(self, branch="main"):
        """Clone the repo if it doesn't exist or is empty."""
        is_empty = not self.target_path.exists() or not any(self.target_path.iterdir())
        
        if is_empty:
            self.target_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning repository to {self.target_path} (branch: {branch})")
            return self._run_command(['git', 'clone', '-b', branch, self.authenticated_url, '.'], cwd=str(self.target_path))
        else:
            logger.info(f"Repository already exists at {self.target_path}. Performing pull.")
            return self.pull()

    def startup_sync(self, branch="main"):
        """Startup routine: sync and fail if error."""
        success, output = self.initialize_repo(branch)
        if not success:
            logger.error(f"Startup Git Sync Failed: {output}")
            raise RuntimeError(f"Startup Git Sync Failed. Container will not start.\n{output}")
        return success, output

    def pull(self):
        """Perform git pull."""
        return self._run_command(['git', 'pull'])

    def push(self, message="Sync from Knowledge Base App", files=["."]):
        """Perform git add, commit, and push."""
        # 1. Add
        success, output = self._run_command(['git', 'add'] + files)
        if not success: return False, output
        
        # 2. Commit
        success, commit_output = self._run_command(['git', 'commit', '-m', message])
        output += "\n" + commit_output
        if not success:
            if "nothing to commit" in commit_output:
                return True, output + "\nNothing to push."
            return False, output
            
        # 3. Push
        success, push_output = self._run_command(['git', 'push'])
        output += "\n" + push_output
        return success, output

    def set_identity(self, name, email):
        """Set git user identity for commits."""
        self._run_command(['git', 'config', 'user.name', name])
        self._run_command(['git', 'config', 'user.email', email])
