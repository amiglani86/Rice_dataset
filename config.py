"""
Configuration for the high-resolution benchmark dataset for rice grains.

All notebooks import this configuration file to access the dataset location and other parameters. 
Edit the 'Local layout' section to point to the location of the dataset on your local machine.
"""

from pathlib import Path

# ==========================
# Local layout
# ==========================
# Expected path:
#
# Data/
#      Original image/
#                     Basmati and non-basmati/
#                                            PB1121/
#                                            PB1401/
#                                            PR14/
#                     BPT5204/
#                             Broken/
#                             Discolored/
#                             Full_chalky/
#                             Healthy/
#                             Partial_Chalky/
#                             Peck_damage/
#                     
#     Segmented image/
#     Train_Val_subset/
# Outputs/
# Checkpoints/

DATA_ROOT = Path("data")
ORIGINAL_IMAGE_DIR = DATA_ROOT / "Original image"
SEGMENTED_IMAGE_DIR = DATA_ROOT / "Segmented image"
SUBSET_DIR = DATA_ROOT / "Train_Val_subset"

OUTPUT_DIR = Path("Outputs")
CHECKPOINT_DIR = Path("Checkpoints")

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ==========================
# Segment anything model
# Download at: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Keep the file sam_vit_b_01ec64.pth in the 'Checkpoints' directory.
# =========================

SAM_CHECKPOINT_PATH = Path(CHECKPOINT_DIR / "sam_vit_b_01ec64.pth")

#HLS Saturation channel range. Pixels outside this range are considered background, including shadows, and will be masked out.
HLS_S_MIN = 0
HLS_S_MAX = 150

# =========================
# Image calibration parameters
# ========================= 
# These calibration parameters are used to convert pixel values to real-world measurements.  
# It is differnt for each dataset and should be set accordingly.

CALIBRATION_BPT = 1.05 
CALIBRATION_BASMATI = 1.79  

CALIBRATION_BPT_mm = CALIBRATION_BPT / 1000  # Convert to mm/pixel
CALIBRATION_BASMATI_mm = CALIBRATION_BASMATI / 1000  # Convert to mm/pixel

MIN_AREA = 100  #px; discard blobs smaller than this during feature extraction

# =========================
# Reproducability
# =========================

SEED = 42  
SEEDS = [42,123,999] # Multi-seed robustness check 


# ===========================
# Deep-learing hyper-parameters  
# ===========================

IMAGE_SIZE    = 224
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 15
NUM_FOLDS     = 5

MAX_IMAGES_PER_CLASS = 1000     # cap per class for the balanced subset
TRAIN_RATIO          = 0.8      # 80:20 train/val split

PCA_COMPONENTS = 50             # PCA before clustering / t-SNE 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
