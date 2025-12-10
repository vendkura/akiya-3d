"""
GitHub Issue Creator for Akiya-3D Thesis
Push issues directly to GitHub repository via API
"""

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class GitHubIssueCreator:
    def __init__(self, repo_owner: str, repo_name: str, token: str = None):
        """
        Initialize GitHub Issue Creator
        
        Args:
            repo_owner: GitHub username or organization
            repo_name: Repository name
            token: GitHub Personal Access Token (or set GITHUB_TOKEN env var)
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
        
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable or pass token parameter.")
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> Dict:
        """
        Create a single GitHub issue
        
        Args:
            title: Issue title
            body: Issue description
            labels: List of label names (optional)
        
        Returns:
            API response with issue details
        """
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {
            'title': title,
            'body': body
        }
        
        if labels:
            data['labels'] = labels
        
        response = requests.post(self.api_url, json=data, headers=headers)
        
        if response.status_code == 201:
            issue = response.json()
            print(f"✅ Created issue #{issue['number']}: {title}")
            print(f"   URL: {issue['html_url']}")
            return issue
        else:
            print(f"❌ Failed to create issue: {title}")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            raise Exception(f"Failed to create issue: {response.status_code}")
    
    def create_issues_batch(self, issues: List[Dict]) -> List[Dict]:
        """
        Create multiple issues
        
        Args:
            issues: List of dicts with 'title', 'body', and optional 'labels'
        
        Returns:
            List of created issue responses
        """
        created = []
        for issue in issues:
            try:
                result = self.create_issue(
                    title=issue['title'],
                    body=issue['body'],
                    labels=issue.get('labels', [])
                )
                created.append(result)
            except Exception as e:
                print(f"⚠️  Skipping issue due to error: {e}")
        
        print(f"\n✅ Successfully created {len(created)}/{len(issues)} issues")
        return created


# Define issues for Experiment 04: FPN vs U-Net work session
ISSUES = [
    {
        'title': '[Experiment 04] Implement FPN architecture training',
        'body': '''**Tasks:**
- [x] Create `fpn-training.py` with ResNet34 encoder + FPN decoder
- [x] Train on 62 images, 13 classes for 50 epochs
- [x] Save model weights and training history

**Files:**
- `U-NET/scripts/fpn-training.py`
- `U-NET/scripts/model_output/fpn_62images/best_model.pth`
- `U-NET/scripts/model_output/fpn_62images/history.json`

**Result:** Mean IoU 53.5% vs U-Net 27.6% (+94% improvement)

See `U-NET/experiment_04_fpn_vs_unet.md` for full details.
''',
        'labels': ['experiment', 'fpn', 'architecture']
    },
    {
        'title': '[Experiment 04] Create per-class analysis tool',
        'body': '''**Tasks:**
- [x] Implement `per_class_analysis_fpn.py`
- [x] Generate per-class IoU, precision, recall, F1 metrics
- [x] Create confusion matrix
- [x] Output detailed statistics report

**Files:**
- `U-NET/scripts/per_class_analysis_fpn.py`
- `U-NET/scripts/model_output/per_class_analysis/fpn_62images_13_classes/`

See `U-NET/experiment_04_fpn_vs_unet.md` for analysis results.
''',
        'labels': ['tooling', 'analysis']
    },
    {
        'title': '[Experiment 04] Document FPN vs U-Net comparison',
        'body': '''**Tasks:**
- [x] Write experiment documentation
- [x] Condense to concise format
- [x] Document architecture decision (adopt FPN)

**Files:**
- `U-NET/experiment_04_fpn_vs_unet.md`

**Decision:** Adopt FPN as primary architecture going forward.
''',
        'labels': ['documentation', 'experiment']
    },
    {
        'title': '[Experiment 05] Consolidate 62+42 image datasets',
        'body': '''**Tasks:**
- [ ] Create dataset merge script (handle filename mismatches)
- [ ] Map 18 categories → 13 unified classes
- [ ] Generate unified masks directory
- [ ] Validate complete 104-image dataset

**Deliverables:**
- `data/floorplan_104/`
- `data/floorplan_masks_104_13classes/`
- `data/annotations/coco-annotation-104images_13classes.json`

**Goal:** +67% more data to reduce overfitting (train-val gap 24%→15%).
''',
        'labels': ['dataset', 'preprocessing', 'priority-high']
    },
    {
        'title': '[Experiment 05] Train FPN on 104-image dataset',
        'body': '''**Tasks:**
- [ ] Update fpn-training.py for 104-image paths
- [ ] Train FPN for 50 epochs (same config as Exp 04)
- [ ] Run per-class analysis
- [ ] Compare vs 62-image baseline
- [ ] Document results

**Depends on:** Issue #4 (dataset consolidation)

**Expected:** Reduce train-val gap to ~15%, maintain/improve 53.5% mIoU.
''',
        'labels': ['experiment', 'fpn', 'training', 'priority-high']
    },
    {
        'title': '[Future] Test higher resolution (1024×1024)',
        'body': '''**Tasks:**
- [ ] Test GPU memory at 1024×1024 (batch_size=1)
- [ ] Implement gradient accumulation if needed
- [ ] Train FPN on 104 images at 1024×1024
- [ ] Compare vs 512×512 baseline

**Expected:** May improve small features (doors/windows) from 35-47% to 50-60% IoU.

**Trade-off:** 2-4× longer training time.
''',
        'labels': ['enhancement', 'future']
    },
    {
        'title': '[Future] Test focal loss for class imbalance',
        'body': '''**Tasks:**
- [ ] Implement focal loss (α=0.25, γ=2)
- [ ] Train and compare vs CrossEntropyLoss
- [ ] Analyze impact on underperforming classes

**Goal:** Balance performance across classes (dining_area 90.6% vs entrance 23.2%).
''',
        'labels': ['enhancement', 'future']
    },
    {
        'title': '[Future] Test deeper encoders (ResNet50/101)',
        'body': '''**Tasks:**
- [ ] Wait for 104-image training (check if train-val gap <15%)
- [ ] Test ResNet50 encoder
- [ ] Compare vs ResNet34 baseline
- [ ] Test ResNet101 only if ResNet50 shows improvement

**Note:** Only proceed if overfitting resolved with more data.
''',
        'labels': ['enhancement', 'future']
    }
]


def main():
    """Create issues for Experiment 04: FPN vs U-Net work"""
    
    # Configuration
    REPO_OWNER = "vendkura"
    REPO_NAME = "akiya-3d"
    
    print("🚀 GitHub Issue Creator - Akiya-3D Thesis")
    print(f"📦 Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"📝 Issues to create: {len(ISSUES)}\n")
    
    # Check for token
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        print("\nTo create a token:")
        print("1. Go to GitHub → Settings → Developer settings → Personal access tokens")
        print("2. Generate new token (classic)")
        print("3. Select scope: 'repo' (full control of private repositories)")
        print("4. Set environment variable: $env:GITHUB_TOKEN='your_token_here'")
        return
    
    # Create issues
    try:
        creator = GitHubIssueCreator(REPO_OWNER, REPO_NAME, token)
        created = creator.create_issues_batch(ISSUES)
        
        print("\n" + "="*60)
        print(f"✅ Successfully created {len(created)} GitHub issues!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
