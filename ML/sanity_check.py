# sanity_check_json.py
import os, json, json
from collections import Counter
from MV_unified_3 import Config, parse_json_to_sequence   # reuse the same parser

config = Config()          # paths already inside
JSON_DIR = config.JSON_DIR
train_split_json = config.DATASET_SPLIT_JSON_PATH

# -------------------------------------------------
# 0.  load the official split (so we only check data the model actually sees)
# -------------------------------------------------
with open(train_split_json, 'r') as f:
    split_data = json.load(f)
train_ids = set(split_data.get('train_ids', []))
print(f"Files in training split: {len(train_ids)}")

# -------------------------------------------------
# 1.  Build a list of all parsed token sequences
# -------------------------------------------------
all_tokens = []
extrude_starts = 0
sketch_before_extrude = 0
duplicate_newbody = 0
broken = []

for fid in train_ids:
    jpath = os.path.join(JSON_DIR, fid + ".json")
    if not os.path.isfile(jpath):
        continue
    try:
        with open(jpath, 'r') as f:
            jdata = json.load(f)
    except Exception as e:
        print("JSON decode error", jpath, e)
        broken.append(jpath)
        continue

    seq = parse_json_to_sequence(jdata)
    all_tokens.extend(seq)

    # -------------------------------------------------
    # 2.  Count extrude blocks
    # -------------------------------------------------
    extrude_idx = [i for i, t in enumerate(seq) if t == "ENTITY_START__Extrude"]
    extrude_starts += len(extrude_idx)

    # -------------------------------------------------
    # 3.  Check whether every extrude has a sketch immediately before
    # -------------------------------------------------
    for ex_idx in extrude_idx:
        if ex_idx == 0:
            print(f"❗ {fid}: extrude starts at position 0 (no sketch possible)")
            broken.append(fid)
            continue
        if seq[ex_idx - 1] != "ENTITY_END__Sketch":
            print(f"❗ {fid}: extrude at pos {ex_idx} not preceded by ENTITY_END__Sketch")
            broken.append(fid)
        else:
            sketch_before_extrude += 1

    # -------------------------------------------------
    # 4.  Count how many sequences have >1  "operation_type=NewBody"
    # -------------------------------------------------
    newbody_count = seq.count("operation_type=NewBody")
    if newbody_count > 1:
        duplicate_newbody += 1
        print(f"❗ {fid}: contains {newbody_count}× NewBody extrude")

# -------------------------------------------------
# 5.  Global token frequency
# -------------------------------------------------
freq = Counter(all_tokens)
print("\nTop-20 most frequent tokens:")
for tok, c in freq.most_common(20):
    print(f"{c:6d}  {tok}")

print("\nSummary")
print("-------")
print(f"Total extrude blocks               : {extrude_starts}")
print(f"Extrudes with sketch before        : {sketch_before_extrude}")
print(f"Files with >1 NewBody extrude      : {duplicate_newbody}")
print(f"Files with structural errors       : {len(set(broken))}")