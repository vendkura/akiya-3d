# 1. First, install everything possible with conda
conda install -c conda-forge pydensecrf shapely svgwrite lmdb fastapi uvicorn tensorboard

# 2. THEN install pip-only packages
pip install segmentation-models-pytorch albumentations mmcv mmdet mmengine svgpathtools tensorboardX
