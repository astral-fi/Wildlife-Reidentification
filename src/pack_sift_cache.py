import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
os.makedirs("cache", exist_ok=True)
train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")

def load_batch(items):
    out = []
    for idx, img_id, folder in items:
        desc = np.load(os.path.join(folder, f"{img_id}_desc.npy"))
        xy = np.load(os.path.join(folder, f"{img_id}_xy.npy"))
        out.append((idx, desc, xy))
    return out

all_items = [(i, img_id, "sift_features_train.npz") for i, img_id in enumerate(train_df["id"].values)] + \
            [(len(train_df) + i, img_id, "sift_features_test.npz") for i, img_id in enumerate(test_df["id"].values)]

chunk_size = 250
chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]

results = [None] * len(all_items)
with ThreadPoolExecutor(max_workers=20) as ex:
    for batch_res in ex.map(load_batch, chunks):
        for idx, desc, xy in batch_res:
            results[idx] = (desc, xy)

offsets = np.zeros(len(all_items) + 1, dtype=np.int64)
for idx, (desc, _) in enumerate(results):
    offsets[idx + 1] = offsets[idx] + desc.shape[0]

total_kp = int(offsets[-1])
all_desc = np.empty((total_kp, 128), dtype=np.uint8)
all_xy = np.empty((total_kp, 2), dtype=np.float32)

for idx, (desc, xy) in enumerate(results):
    s, e = offsets[idx], offsets[idx + 1]
    if e > s:
        all_desc[s:e] = desc
        all_xy[s:e] = xy

np.save("cache/sift_offsets.npy", offsets)
np.save("cache/sift_desc.npy", all_desc)
np.save("cache/sift_xy.npy", all_xy)
print(f"Packed {len(all_items)} images ({total_kp:,} keypoints) into cache/ in {time.time() - t0:.1f}s")
