import os, glob

img_dir = r"E:\road_dataset\relabel\val\images"
lbl_dir = r"E:\road_dataset\relabel\val\labels"

os.makedirs(lbl_dir, exist_ok=True)

imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
lbls = set(os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(lbl_dir, "*.txt")))

missing = [os.path.splitext(os.path.basename(x))[0] for x in imgs if os.path.splitext(os.path.basename(x))[0] not in lbls]

for stem in missing:
    open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8").close()

print("Created empty VAL labels:", len(missing))
print("First examples:", missing[:20])
