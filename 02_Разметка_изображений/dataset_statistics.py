import glob
from collections import Counter

def stats(labels_glob):
    img_has = Counter()
    inst = Counter()
    files = glob.glob(labels_glob)
    for p in files:
        present = set()
        for ln in open(p, "r", encoding="utf-8", errors="ignore"):
            s = ln.strip().split()
            if len(s) != 5:
                continue
            c = int(float(s[0]))
            present.add(c)
            inst[c] += 1
        for c in present:
            img_has[c] += 1
    return len(files), img_has, inst

train_n, train_has, train_inst = stats(r"E:\road_dataset\relabel\train\labels\*.txt")
val_n, val_has, val_inst = stats(r"E:\road_dataset\relabel\val\labels\*.txt")

print("TRAIN label files:", train_n)
print("TRAIN images_with_class:", dict(train_has))
print("TRAIN instances:", dict(train_inst))
print()
print("VAL label files:", val_n)
print("VAL images_with_class:", dict(val_has))
print("VAL instances:", dict(val_inst))
