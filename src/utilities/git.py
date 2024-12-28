import subprocess

def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print("Error while retrieving git commit hash:", e)
        return None