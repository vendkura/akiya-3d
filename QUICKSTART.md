# Quick Start Guide - 3D Floor Plan Web Demo

## TL;DR - 30 Seconds to Running

```bash
# 1. Go to web demo directory
cd web_demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Flask server
python app.py

# 4. Open browser
http://localhost:5000
```

Done! Upload a floor plan image and watch it convert to 3D.

---

## Detailed Setup (5 minutes)

### Prerequisites Check
```bash
# Verify Python 3.8+
python --version

# Verify conda environment is active
conda info --envs
# Should show (akya3d) in terminal prompt
```

### Step 1: Navigate to Web Demo
```bash
cd e:\github.com\akiya-3d-thesis\web_demo
# Or: cd /mnt/e/github.com/akiya-3d-thesis/web_demo (WSL/Linux)
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed flask-2.3.3 flask-cors-4.0.0 torch-2.0.1 ...
```

### Step 3: Start Flask Server
```bash
python app.py
```

**Expected output:**
```
Starting Flask web demo server...
Upload folder: [...]/uploads
Output folder: [...]/outputs
 * Running on http://localhost:5000
 * Press CTRL+C to quit
```

### Step 4: Open in Browser
```
http://localhost:5000
```

You should see a beautiful purple interface with "3D Floor Plan Reconstruction" header.

---

## Using the Web Demo

### Simple Workflow
1. **Drag & Drop** - Drag a floor plan PNG/JPG onto the upload area
   - Or click to browse and select file
   - Supported: `.png`, `.jpg`, `.jpeg`
   - Max size: 10 MB

2. **Process** - Click blue "Process Image" button
   - Wait for processing (8-15 seconds on CPU)
   - Watch status update in real-time

3. **Download** - Once complete, download files
   - Click green "Download" buttons for OBJ and MTL
   - Save to your computer

4. **View** - Open OBJ file in 3D viewer
   - Online: [ViewSTL.com](https://www.viewstl.com/) or Sketchfab
   - Desktop: Blender, Meshlab, etc.

### Example Test Image
```
Located at: ../U-NET/data/floorplan/
Filename: Capture d'écran 2025-06-05 224143.png

Or use any other floor plan PNG in that directory
```

---

## Troubleshooting

### "Module not found" Error
```
ModuleNotFoundError: No module named 'torch'
```
**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

### "Model not found" Error
```
FileNotFoundError: Model not found at ../U-NET/scripts/model_output/fpn_62images/best_model.pth
```
**Fix:** Verify FPN model exists:
```bash
ls ../U-NET/scripts/model_output/fpn_62images/best_model.pth
```
If missing, update model path in `app.py` line 54

### "Address already in use" Error
```
OSError: [Errno 48] Address already in use
```
**Fix:** Use different port:
```bash
# Edit app.py, change last line to:
app.run(host='localhost', port=5001, debug=True)
```
Then visit: `http://localhost:5001`

### Upload Fails
- Check file size < 10 MB
- Verify file is PNG or JPG
- Check `uploads/` folder has write permissions
- Clear browser cache (Ctrl+Shift+Del)

### Processing Hangs
- Check terminal for errors (scroll up)
- Verify FPN model loaded successfully
- Check available RAM (4GB+ needed)
- Try CPU mode (automatic if CUDA unavailable)

---

## API Usage (For Developers)

### Direct Python Usage
```python
from pathlib import Path
import sys

# Add pipeline to path
sys.path.insert(0, '../3D-Pipeline')
from main_pipeline import Pipeline3D

# Initialize
pipeline = Pipeline3D()

# Process image
result = pipeline.process_image('floor_plan.png')

# Check result
if result['status'] == 'success':
    print(f"✓ OBJ: {result['obj_file']}")
    print(f"✓ MTL: {result['mtl_file']}")
    
    # Access metadata
    metadata = result['metadata']
    print(f"Vertices: {metadata['steps']['wall_extrusion']['num_vertices']}")
    print(f"Faces: {metadata['steps']['wall_extrusion']['num_faces']}")
else:
    print(f"✗ Error: {result['error']}")
```

### REST API Usage
```bash
# 1. Upload file
curl -X POST -F "file=@floor_plan.png" http://localhost:5000/api/upload

# Response: { "task_id": "20240101_120000_floorplan", ... }

# 2. Process image
curl -X POST http://localhost:5000/api/process/20240101_120000_floorplan

# 3. Download OBJ
curl -X GET http://localhost:5000/api/download/20240101_120000_floorplan/obj \
  --output floor_plan.obj

# 4. Download MTL
curl -X GET http://localhost:5000/api/download/20240101_120000_floorplan/mtl \
  --output floor_plan.mtl
```

---

## Viewing Generated Models

### Quick Online Viewers
- **ViewSTL.com** - Drag & drop OBJ file
- **Sketchfab** - Upload & share online
- **Three.js Editor** - Advanced viewer
- **Model Viewer** - Google's web component

### Desktop Software
- **Blender** - Free, professional
- **Meshlab** - Free, lightweight
- **3DS Max** - Paid, professional
- **Maya** - Paid, professional

### Python/Code
```python
import open3d as o3d

# Load and view
mesh = o3d.io.read_triangle_mesh("floor_plan.obj")
o3d.visualization.draw_geometries([mesh])
```

---

## File Organization

```
web_demo/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── README.md          # Full documentation
├── templates/
│   └── index.html     # Web interface
├── uploads/           # Uploaded images
├── outputs/           # Generated OBJ/MTL
└── static/           # CSS/JS (optional)
```

---

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| PyTorch | 2.0.1 |
| Flask | 2.3.3 |
| OpenCV | 4.8.0.76 |
| NumPy | 1.24.3 |

**Conda Environment:** `akya3d`

---

## Performance Tips

### Faster Processing (GPU)
```bash
# CUDA will be auto-detected
# Requires NVIDIA GPU + CUDA toolkit
# ~5-10x faster than CPU
```

### Batch Processing
```python
# Process multiple images
results = pipeline.process_batch('image_folder/')
```

### Reduce Processing Time
- Smaller input images (256×256 instead of 512×512)
- Lower quality requirements
- Disable intermediate JSON outputs

---

## Next Steps

✅ **Setup Complete?** Run some test images!

**What to test:**
1. Simple floor plans (1-5 rooms)
2. Complex layouts (10+ rooms)
3. Different image formats (PNG, JPG)
4. Various resolutions (512×512 to 2048×2048)

**Things to verify:**
- OBJ files open in 3D viewer
- MTL materials load correctly
- All rooms are visible
- Geometry is valid (no self-intersections)

---

## Getting Help

**Check these first:**
1. Terminal output - scroll up for error messages
2. Browser console - F12 → Console tab
3. `web_demo/README.md` - Full documentation
4. `3D-Pipeline/main_pipeline.py` - Code docstrings

**Common files to check:**
- `app.py` line 54 - FPN model path
- `web_demo/requirements.txt` - Dependencies
- `../U-NET/scripts/model_output/fpn_62images/best_model.pth` - Model location

---

## Important Notes

⚠️ **First Run:** FPN model loads on first request (takes ~30 seconds)

⚠️ **CPU Mode:** Default, slower but works on any computer

⚠️ **File Cleanup:** Old uploads in `uploads/` folder can be deleted

⚠️ **Output Size:** Generated OBJ files ~100-200 KB each

---

**Happy modeling! 🎉**

Questions? Check `web_demo/README.md` for detailed documentation.
