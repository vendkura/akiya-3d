import requests
import json
import os

# GitHub Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')  # Set this as environment variable
REPO_OWNER = "vendkura"
REPO_NAME = "akiya-3d"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

base_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"

# Issues to create - Experiment 05 work session
issues = [
    {
        "title": "[Experiment 05] Consolidate 62+42 image datasets",
        "body": """**Tasks:**
- [x] Create `simple_consolidate_104.py` for initial merge
- [x] Identify missing room annotations in new 42 images
- [x] Manually annotate rooms with AI assistance (4+ floor plans)
- [x] Create `final_consolidate_104.py` with complete annotations
- [x] Handle filename mismatches (hash prefixes, URL encoding)
- [x] Map 18 categories → 13 unified classes (DK→room, toilet→bathroom, etc.)
- [x] Generate 104 complete image-mask pairs

**Files:**
- `U-NET/scripts/simple_consolidate_104.py`
- `U-NET/scripts/final_consolidate_104.py`
- `U-NET/data/floorplan_104_final/` (104 images)
- `U-NET/data/floorplan_masks_104_13classes_final/` (104 masks)
- `U-NET/data/annotations/new-42-images-annotations.json`

**Result:** Successfully merged 62 original + 42 newly annotated images.
""",
        "labels": ["dataset", "preprocessing", "experiment"]
    },
    {
        "title": "[Experiment 05] Train FPN on 104-image dataset",
        "body": """**Tasks:**
- [x] Update `fpn-training.py` for 104-image paths
- [x] Train FPN for 50 epochs (same config as Exp 04)
- [x] Train twice: incomplete annotations (35.6% val IoU), then complete (32.4% val IoU)
- [x] Run per-class analysis on both 104-image models
- [x] Compare vs 62-image baseline

**Files:**
- `U-NET/scripts/fpn-training.py` (updated for 104 images)
- `U-NET/scripts/model_output/fpn_104images/` (incomplete annotations)
- `U-NET/scripts/model_output/fpn_104images_final/` (complete annotations)
- `U-NET/scripts/model_output/per_class_analysis/fpn_104images_13_classes/`
- `U-NET/scripts/model_output/per_class_analysis/fpn_104images_final_13_classes/`

**Result:** 104 complete model achieved 47.8% mean IoU, 32.4% val IoU.
""",
        "labels": ["experiment", "fpn", "training"]
    },
    {
        "title": "[Experiment 05] Document Dataset Size Impact Study",
        "body": """**Tasks:**
- [x] Compare 62 vs 104 image performance
- [x] Analyze per-class results (11/12 classes favor 62 images)
- [x] Identify root cause: data quality > quantity
- [x] Write concise experiment report
- [x] Document production model decision

**Files:**
- `U-NET/experiment_05_dataset_size.md`

**Key Finding:** **Quality > Quantity**
- 62 images: 53.5% mean IoU ✅ **BEST**
- 104 complete: 47.8% mean IoU
- 62-image model wins in 11 out of 12 classes

**Decision:** Use 62-image FPN model for production.
""",
        "labels": ["documentation", "experiment", "completed"]
    },
    {
        "title": "[Future] Repository cleanup and 3D pipeline",
        "body": """**Next work session tasks:**
- [ ] Remove experimental 104-image files
- [ ] Organize final directory structure
- [ ] Implement 3D boundary extraction
- [ ] Implement wall extrusion + OBJ export
- [ ] Build web demo
- [ ] Conduct user testing (5-10 users)
- [ ] Complete thesis documentation

**Timeline:** 1 month (Dec 16 - Jan 16, 2026)

**Production Model:** 62-image FPN (53.5% mean IoU)
""",
        "labels": ["future", "3d-pipeline", "cleanup"]
    }
]

def create_issues():
    created_issues = []
    
    for issue_data in issues:
        print(f"\nCreating issue: {issue_data['title']}")
        
        response = requests.post(
            base_url,
            headers=headers,
            data=json.dumps(issue_data)
        )
        
        if response.status_code == 201:
            issue = response.json()
            created_issues.append({
                'number': issue['number'],
                'title': issue['title'],
                'url': issue['html_url']
            })
            print(f"✅ Created issue #{issue['number']}: {issue['title']}")
        else:
            print(f"❌ Failed to create issue: {issue_data['title']}")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
    
    return created_issues

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        print("\nTo set it:")
        print('PowerShell: $env:GITHUB_TOKEN="your_token_here"')
        print('Then run: python create_experiment_05_issues.py')
    else:
        print(f"Creating issues for {REPO_OWNER}/{REPO_NAME}...")
        created = create_issues()
        
        print("\n" + "="*60)
        print(f"Summary: Created {len(created)} issues")
        print("="*60)
        
        for issue in created:
            print(f"#{issue['number']}: {issue['title']}")
            print(f"  → {issue['url']}")
