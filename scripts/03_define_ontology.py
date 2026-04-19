# This is the master ontology for your project
# Based on: MOCS (13 categories) + ConstructionSite10k (PPE safety)

MOCS_CLASSES = [
    "worker",
    "tower crane",
    "hanging hook",
    "truck crane",
    "roller",
    "bulldozer",
    "excavator",
    "truck",
    "loader",
    "pump truck",
    "concrete truck",
    "pile driver",
    "other vehicle"
]

PPE_SAFETY_CLASSES = [
    "helmet",
    "hard hat",
    "safety vest",
    "boots",
    "gloves",
    "goggles",
    "scaffold",
    "ladder",
    "rebar",
    "trench",
    "debris",
    "missing helmet",
    "missing vest"
]

# Combined ontology for your GroundingDINO prompts
FULL_ONTOLOGY = MOCS_CLASSES + PPE_SAFETY_CLASSES

# For YOLOv8 training (use a cleaner subset)
YOLO_CLASSES = [
    "worker", "excavator", "truck", "loader", "bulldozer",
    "tower crane", "roller", "helmet", "safety vest",
    "scaffold", "ladder", "concrete truck", "pump truck"
]

print(f"Full ontology: {len(FULL_ONTOLOGY)} classes")
print(f"YOLO training classes: {len(YOLO_CLASSES)} classes")

import json, os
os.makedirs("/Data1/cse_24203016/construction_site/scripts", exist_ok=True)
with open("/Data1/cse_24203016/construction_site/scripts/ontology.json", "w") as f:
    json.dump({
        "mocs_classes": MOCS_CLASSES,
        "ppe_classes": PPE_SAFETY_CLASSES,
        "full_ontology": FULL_ONTOLOGY,
        "yolo_classes": YOLO_CLASSES
    }, f, indent=2)

print("Saved ? scripts/ontology.json")