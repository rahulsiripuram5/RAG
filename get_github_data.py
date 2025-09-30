import os
from github import Github, Auth, RateLimitExceededException
import time

# --- CONFIGURATION ---
GITHUB_TOKEN = "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" 
REPO_NAME = "microsoft/vscode"
DATA_DIR = "data_github"
ISSUES_TO_FETCH = 100 

# --- SCRIPT ---

os.makedirs(DATA_DIR, exist_ok=True)

# 1. AUTHENTICATION (The New Way)
# Use the Auth.Token method as recommended by the warning
auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)

print(f"Fetching issues from {REPO_NAME}...")

try:
    repo = g.get_repo(REPO_NAME)
    issues = repo.get_issues(state='closed', sort='updated', direction='desc')
    
    count = 0
    for issue in issues:
        if count >= ISSUES_TO_FETCH:
            break
        
        if issue.pull_request:
            continue
            
        issue_content = f"Issue #{issue.number}: {issue.title}\n\n"
        issue_content += f"State: {issue.state}\n"
        issue_content += f"Author: {issue.user.login}\n\n"
        
        if issue.body:
            issue_content += "--- ISSUE BODY ---\n"
            issue_content += issue.body + "\n\n"
            
        comments = issue.get_comments()
        if comments.totalCount > 0:
            issue_content += "--- COMMENTS ---\n"
            for comment in comments:
                issue_content += f"Comment by {comment.user.login}:\n{comment.body}\n\n"
        
        sanitized_title = "".join(c for c in issue.title if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = f"issue_{issue.number}_{sanitized_title[:50]}.txt"
        file_path = os.path.join(DATA_DIR, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(issue_content)
        
        count += 1
        print(f"({count}/{ISSUES_TO_FETCH}) Saved issue #{issue.number}")
        
        # 2. RATE LIMIT CHECK (The New Way)
        # The remaining requests are now at g.rate_limiting[0]
        remaining_requests = g.rate_limiting[0]
        if remaining_requests < 20:
            print(f"Approaching rate limit ({remaining_requests} left), sleeping for 60 seconds...")
            time.sleep(60)

    print(f"\nSuccessfully fetched and saved {count} issues to the '{DATA_DIR}' directory.")

except RateLimitExceededException:
    print("GitHub API rate limit exceeded. Please wait a while or use a valid PAT.")
except Exception as e:
    print(f"An error occurred: {e}")