import os, glob

val_imgs = glob.glob(r"E:\road_dataset\relabel\val\images\*.jpg")
val_lbls = set(os.path.splitext(os.path.basename(x))[0] for x in glob.glob(r"E:\road_dataset\relabel\val\labels\*.txt"))

missing = [os.path.basename(x) for x in val_imgs if os.path.splitext(os.path.basename(x))[0] not in val_lbls]

print("VAL images:", len(val_imgs))
print("VAL labels:", len(val_lbls))
print("VAL missing labels:", len(missing))
print("First missing examples:", missing[:20])
