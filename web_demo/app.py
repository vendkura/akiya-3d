"""
Flask Web Demo for 3D Floor Plan Reconstruction
API endpoints for uploading floor plans and generating 3D models
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pathlib import Path
import json
import os
import sys
import logging
from datetime import datetime

# Add 3D-Pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "3D-Pipeline"))

from main_pipeline import Pipeline3D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
OUTPUT_FOLDER = Path(__file__).parent / 'outputs'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Initialize pipeline
try:
    pipeline = Pipeline3D(
        fpn_model_path="../U-NET/scripts/model_output/fpn_62images/best_model.pth",
        output_dir=str(OUTPUT_FOLDER)
    )
    logger.info("✓ Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None


# ===================== ROUTES =====================

@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'pipeline_ready': pipeline is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload a floor plan image and start processing.
    
    Returns:
        {
            'status': 'success' | 'error',
            'task_id': str,
            'message': str,
            'error': str (if error)
        }
    """
    try:
        # Check if pipeline is ready
        if pipeline is None:
            return jsonify({
                'status': 'error',
                'message': 'Pipeline not initialized. Check server logs.'
            }), 500
        
        # Check if file in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Empty filename'
            }), 400
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB'
            }), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOAD_FOLDER / filename
        
        file.save(str(filepath))
        
        # Create task ID from filename
        task_id = filepath.stem
        
        logger.info(f"✓ Uploaded: {filename}")
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': f'File uploaded: {filename}. Processing started...'
        }), 200
    
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/process/<task_id>', methods=['POST'])
def process(task_id):
    """
    Process an uploaded image and generate 3D model.
    
    Args:
        task_id: ID from upload endpoint
    
    Returns:
        {
            'status': 'success' | 'error',
            'task_id': str,
            'obj_file': str (path to OBJ),
            'mtl_file': str (path to MTL),
            'metadata': dict,
            'message': str,
            'error': str (if error)
        }
    """
    try:
        # Find uploaded image
        uploaded_files = list(UPLOAD_FOLDER.glob(f"{task_id}*"))
        
        if not uploaded_files:
            return jsonify({
                'status': 'error',
                'message': f'Task {task_id} not found'
            }), 404
        
        image_path = uploaded_files[0]
        
        logger.info(f"Processing task: {task_id}")
        
        # Run pipeline
        result = pipeline.process_image(
            str(image_path),
            output_name=task_id,
            save_intermediate=False
        )
        
        if result['status'] == 'error':
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Processing failed'),
                'task_id': task_id
            }), 500
        
        logger.info(f"✓ Processing complete: {task_id}")
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'obj_file': Path(result['obj_file']).name,
            'mtl_file': Path(result['mtl_file']).name,
            'message': 'Processing complete!',
            'metadata': result['metadata']
        }), 200
    
    except Exception as e:
        logger.error(f"Process error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'task_id': task_id
        }), 500


@app.route('/api/download/<task_id>/<filetype>', methods=['GET'])
def download(task_id, filetype):
    """
    Download OBJ or MTL file.
    
    Args:
        task_id: Task ID
        filetype: 'obj' or 'mtl'
    
    Returns:
        File content with appropriate headers
    """
    try:
        if filetype not in ['obj', 'mtl']:
            return jsonify({'error': 'Invalid filetype'}), 400
        
        # Find output file
        output_dir = OUTPUT_FOLDER / task_id
        
        if filetype == 'obj':
            filepath = output_dir / f"{task_id}.obj"
        else:
            filepath = output_dir / f"{task_id}.mtl"
        
        if not filepath.exists():
            return jsonify({'error': f'{filetype.upper()} file not found'}), 404
        
        logger.info(f"Downloading {filetype}: {filepath.name}")
        
        return send_file(
            str(filepath),
            as_attachment=True,
            download_name=filepath.name,
            mimetype='text/plain'
        )
    
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/<task_id>', methods=['GET'])
def preview(task_id):
    """
    Get 3D model data for preview.
    
    Returns:
        {
            'status': 'success',
            'vertices': [[x, y, z], ...],
            'faces': [[v1, v2, v3], ...],
            'colors': [[r, g, b], ...],
            'message': str
        }
    """
    try:
        # Check if geometry exists
        output_dir = OUTPUT_FOLDER / task_id
        geometry_path = output_dir / f"{task_id}_geometry.json"
        
        # Try to find metadata with geometry
        metadata_path = output_dir / f"{task_id}_metadata.json"
        
        if not metadata_path.exists():
            return jsonify({
                'status': 'error',
                'message': 'No preview data available'
            }), 404
        
        # We'll need to regenerate or store the geometry
        # For now, return a message that OBJ file is ready
        obj_file = output_dir / f"{task_id}.obj"
        
        if not obj_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'OBJ file not found'
            }), 404
        
        # Read OBJ file to extract vertices and faces
        vertices = []
        faces = []
        
        with open(obj_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(x) for x in parts[1:4]])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(face)
        
        logger.info(f"Preview: {len(vertices)} vertices, {len(faces)} faces")
        
        return jsonify({
            'status': 'success',
            'vertices': vertices[:1000],  # Limit for preview
            'faces': faces[:1000],
            'message': f'{len(vertices)} vertices, {len(faces)} faces'
        }), 200
    
    except Exception as e:
        logger.error(f"Preview error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def status(task_id):
    """
    Check processing status.
    
    Returns:
        {
            'status': 'processing' | 'complete' | 'failed',
            'task_id': str,
            'message': str
        }
    """
    try:
        output_dir = OUTPUT_FOLDER / task_id
        obj_file = output_dir / f"{task_id}.obj"
        
        if obj_file.exists():
            return jsonify({
                'status': 'complete',
                'task_id': task_id,
                'message': 'Ready for download'
            }), 200
        
        uploaded_file = list(UPLOAD_FOLDER.glob(f"{task_id}*"))
        
        if uploaded_file:
            return jsonify({
                'status': 'processing',
                'task_id': task_id,
                'message': 'Processing in progress...'
            }), 200
        
        return jsonify({
            'status': 'not_found',
            'task_id': task_id,
            'message': 'Task not found'
        }), 404
    
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500


# ===================== MAIN =====================

if __name__ == '__main__':
    logger.info("Starting Flask web demo server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    
    # Run development server
    app.run(
        host='localhost',
        port=5000,
        debug=True,
        use_reloader=False  # Prevent reloading during file uploads
    )
