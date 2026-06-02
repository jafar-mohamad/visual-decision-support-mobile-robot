import os
import random
import shutil

BASE = r"E:\road_dataset"
SRC = os.path.join(BASE, "images_all")
TRAIN_IMG = os.path.join(BASE, "train", "images")
VAL_IMG = os.path.join(BASE, "val", "images")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)

images = [f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

n_total = len(images)
n_train = int(n_total * 0.8)

train_files = images[:n_train]
val_files = images[n_train:]

for f in train_files:
    src_path = os.path.join(SRC, f)
    dst_path = os.path.join(TRAIN_IMG, f)
    shutil.copy2(src_path, dst_path)

for f in val_files:
    src_path = os.path.join(SRC, f)
    dst_path = os.path.join(VAL_IMG, f)
    shutil.copy2(src_path, dst_path)

print("Total images:", n_total)
print("Train images:", len(train_files))
print("Val images:", len(val_files))
