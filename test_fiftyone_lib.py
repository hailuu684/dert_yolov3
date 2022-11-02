import fiftyone

"""
Works in computer in lab but takes too much memories. appx 144Gb
"""
dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "car"],
    max_samples=50,
)