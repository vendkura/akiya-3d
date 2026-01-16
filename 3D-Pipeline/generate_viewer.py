"""
Interactive 3D Visualization of Generated OBJ Models
Creates an HTML file with Three.js viewer for interactive inspection.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


def generate_three_js_viewer(obj_path: str, output_html: str):
    """
    Generate an interactive Three.js HTML viewer for OBJ file.
    
    Args:
        obj_path: Path to OBJ file
        output_html: Output HTML file path
    """
    obj_path = Path(obj_path)
    
    if not obj_path.exists():
        print(f"OBJ file not found: {obj_path}")
        return
    
    # Read OBJ file
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    face.append(int(indices[0]) - 1)
                if len(face) == 3:
                    faces.append(face)
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Calculate bounds
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    center = (min_v + max_v) / 2
    size = max_v - min_v
    
    print(f"✓ Loaded: {len(vertices)} vertices, {len(faces)} faces")
    print(f"✓ Bounds: {min_v.tolist()} to {max_v.tolist()}")
    
    # Create HTML with Three.js
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Floor Plan Viewer</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background: #1a1a1a;
            font-family: Arial, sans-serif;
        }}
        
        canvas {{
            display: block;
            width: 100%;
            height: 100vh;
        }}
        
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: #fff;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 5px;
            font-size: 14px;
            max-width: 300px;
        }}
        
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: #fff;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
        }}
        
        .control-group {{
            margin-bottom: 10px;
        }}
        
        .control-label {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        
        button:hover {{
            background: #764ba2;
        }}
        
        #stats {{
            position: absolute;
            top: 10px;
            right: 10px;
            color: #fff;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h3>3D Floor Plan</h3>
        <p><strong>File:</strong> {obj_path.name}</p>
        <p><strong>Vertices:</strong> {len(vertices)}</p>
        <p><strong>Faces:</strong> {len(faces)}</p>
        <p><strong>Size:</strong> {size[0]:.1f}m × {size[1]:.1f}m × {size[2]:.1f}m</p>
    </div>
    
    <div id="controls">
        <div class="control-group">
            <div class="control-label">🎮 Controls</div>
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleWireframe()">Wireframe</button>
            <button onclick="toggleLights()">Lights</button>
            <br>
            <button onclick="rotateAuto()">Auto Rotate</button>
        </div>
        <div class="control-group">
            <div class="control-label">💾 Export</div>
            <button onclick="downloadOBJ()">Download OBJ</button>
        </div>
    </div>
    
    <div id="stats">
        FPS: <span id="fps">60</span><br>
        Rotation: <span id="rotation">0°</span>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2a2a2a);
        
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        
        const verticesArray = {json.dumps(vertices.tolist())};
        const facesArray = {json.dumps(faces.tolist())};
        
        const positions = new Float32Array(verticesArray.flat());
        const indices = new Uint32Array(facesArray.flat());
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();
        
        // Material and mesh
        const material = new THREE.MeshPhongMaterial({{
            color: 0x667eea,
            shininess: 100,
            side: THREE.DoubleSide,
            wireframe: false
        }});
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        
        // Ground plane
        const groundGeometry = new THREE.PlaneGeometry(100, 100);
        const groundMaterial = new THREE.MeshStandardMaterial({{ color: 0x444444 }});
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.1;
        ground.receiveShadow = true;
        scene.add(ground);
        
        // Camera position
        const center = {json.dumps(center.tolist())};
        const maxDim = Math.max({size[0]}, {size[1]}, {size[2]});
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.5;
        
        camera.position.set(center[0], center[1] + maxDim * 0.5, center[2] + cameraZ);
        camera.lookAt(center[0], center[1], center[2]);
        
        // Mouse controls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let targetRotationX = 0;
        let targetRotationY = 0;
        
        document.addEventListener('mousedown', (e) => {{
            mouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (!mouseDown) return;
            
            const deltaX = e.clientX - mouseX;
            const deltaY = e.clientY - mouseY;
            
            targetRotationY += deltaX * 0.005;
            targetRotationX += deltaY * 0.005;
            
            mouseX = e.clientX;
            mouseY = e.clientY;
        }});
        
        document.addEventListener('mouseup', () => {{
            mouseDown = false;
        }});
        
        // Zoom with scroll
        document.addEventListener('wheel', (e) => {{
            e.preventDefault();
            camera.position.multiplyScalar(1 + e.deltaY * 0.001);
        }}, {{ passive: false }});
        
        // Variables
        let wireframeMode = false;
        let autoRotate = false;
        let lastTime = Date.now();
        let frameCount = 0;
        
        // Functions
        function resetView() {{
            targetRotationX = 0;
            targetRotationY = 0;
            mesh.rotation.x = 0;
            mesh.rotation.y = 0;
            camera.position.set(center[0], center[1] + maxDim * 0.5, center[2] + cameraZ);
            camera.lookAt(center[0], center[1], center[2]);
        }}
        
        function toggleWireframe() {{
            wireframeMode = !wireframeMode;
            material.wireframe = wireframeMode;
        }}
        
        function toggleLights() {{
            ambientLight.intensity = ambientLight.intensity > 0.3 ? 0.3 : 0.6;
            directionalLight.intensity = directionalLight.intensity > 0.4 ? 0.4 : 0.8;
        }}
        
        function rotateAuto() {{
            autoRotate = !autoRotate;
        }}
        
        function downloadOBJ() {{
            alert('OBJ file is already available for download from the web interface.');
        }}
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            
            // Auto rotation
            if (autoRotate) {{
                mesh.rotation.y += 0.005;
            }} else {{
                mesh.rotation.x += (targetRotationX - mesh.rotation.x) * 0.1;
                mesh.rotation.y += (targetRotationY - mesh.rotation.y) * 0.1;
            }}
            
            // Update stats
            frameCount++;
            const now = Date.now();
            if (now - lastTime >= 1000) {{
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }}
            
            const degrees = (mesh.rotation.y * 180 / Math.PI) % 360;
            document.getElementById('rotation').textContent = degrees.toFixed(0) + '°';
            
            renderer.render(scene, camera);
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_html)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Generated viewer: {output_path}")
    print(f"✓ Open in browser to view 3D model")
    print(f"\n  Controls:")
    print(f"  - Drag mouse: Rotate")
    print(f"  - Scroll: Zoom")
    print(f"  - Click buttons: Toggle features")


if __name__ == "__main__":
    obj_path = Path("pipeline_output/Capture d’écran 2025-06-05 224143/Capture d’écran 2025-06-05 224143.obj")
    output_html = Path("pipeline_output/Capture d’écran 2025-06-05 224143/viewer_3d.html")
    
    if obj_path.exists():
        generate_three_js_viewer(str(obj_path), str(output_html))
    else:
        print(f"OBJ file not found: {obj_path}")
        print("Run main_pipeline.py first")
