# -*- coding: utf-8 -*-
"""COCO (实例分割) -> YOLO bbox 数据集转换。

用法: python analysis/coco_to_yolo.py <zip路径或解压目录>
- 自动解压(如为 zip)
- 读取 COCO JSON, 输出 YOLO 格式 train/valid/test 划分到 dataset/
- 打印类别映射与数据统计
"""
import os
import sys
import json
import zipfile
import shutil
import random
import glob
from collections import Counter

random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
SPLITS = {"train": 0.7, "valid": 0.2, "test": 0.1}


def find_coco_jsons(root):
    hits = []
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if "archive" in p or "dataset_info" in p:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "images" in d and "annotations" in d:
                hits.append(p)
        except Exception:
            pass
    return hits


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("TEMP", ""), "blueberry_hf", "berry_coco-formant-annotation.zip")
    work = os.path.join(os.path.dirname(src), "blueberry_extract")
    if os.path.isdir(work):
        shutil.rmtree(work)
    if zipfile.is_zipfile(src):
        print("unzipping...")
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
        print("done ->", work)
    else:
        work = src

    jsons = find_coco_jsons(work)
    if not jsons:
        print("no COCO json found under", work)
        sys.exit(1)
    print("COCO jsons:", jsons)
    ann_path = sorted(jsons, key=os.path.getsize)[-1]  # 最大的那个
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    print("categories:", cat_map)
    # COCO id -> 0-based YOLO id
    sorted_ids = sorted(cat_map.keys())
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    yolo_names = [cat_map[i] for i in sorted_ids]
    print("yolo names:", yolo_names)
    img_info = {i["id"]: i for i in coco["images"]}
    print("images:", len(coco["images"]), "annotations:", len(coco["annotations"]))

    # 逐图收集目标
    per_img = {}
    for a in coco["annotations"]:
        iid = a["image_id"]
        per_img.setdefault(iid, []).append(a)
    used_imgs = sorted(per_img.keys())
    random.shuffle(used_imgs)

    n = len(used_imgs)
    n_train = int(n * SPLITS["train"])
    n_valid = int(n * SPLITS["valid"])
    assigns = {}
    for i, iid in enumerate(used_imgs):
        if i < n_train:
            assigns[iid] = "train"
        elif i < n_train + n_valid:
            assigns[iid] = "valid"
        else:
            assigns[iid] = "test"

    # 清理旧数据集目录(保留 data.yaml 由本脚本重写)
    for s in SPLITS:
        for sub in ("images", "labels"):
            p = os.path.join(OUT, s, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p)

    cls_count = Counter()
    empty = Counter()
    total_boxes = 0
    for iid, split in assigns.items():
        info = img_info[iid]
        src_img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = os.path.join(os.path.dirname(ann_path), info["file_name"])
            if os.path.exists(cand):
                src_img = cand
                break
            cand = os.path.join(work, info["file_name"].split("/")[-1])
            if os.path.exists(cand):
                src_img = cand
                break
        if src_img is None:
            # 递归查找
            base = os.path.basename(info["file_name"])
            hits = glob.glob(os.path.join(work, "**", base), recursive=True)
            src_img = hits[0] if hits else None
        if src_img is None:
            print("  [missing image]", info["file_name"])
            continue

        stem = os.path.splitext(os.path.basename(src_img))[0]
        w, h = info["width"], info["height"]
        new_name = f"{iid}_{os.path.basename(src_img)}"   # 防跨子集同名覆盖
        new_stem = os.path.splitext(new_name)[0]
        shutil.copy(src_img, os.path.join(OUT, split, "images", new_name))

        anns = per_img.get(iid, [])
        lines = []
        for a in anns:
            cid = a["category_id"]
            if cid not in cat_map:
                continue
            yid = id_map[cid]
            # 优先用 COCO bbox, 否则从分割掩码计算
            if "bbox" in a and a["bbox"]:
                x, y, bw, bh = a["bbox"]
            else:
                xs = [p[0] for seg in a["segmentation"] for p in zip(seg[0::2], seg[1::2])]
                ys = [p[1] for seg in a["segmentation"] for p in zip(seg[0::2], seg[1::2])]
                x, y = min(xs), min(ys)
                bw, bh = max(xs) - x, max(ys) - y
            cx, cy = x + bw / 2, y + bh / 2
            lines.append(f"{yid} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            cls_count[yid] += 1
            total_boxes += 1
        if not lines:
            empty[split] += 1
        with open(os.path.join(OUT, split, "labels", new_stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    print("-" * 60)
    print("类别 -> 目标数 (YOLO id):")
    for cid, name in cat_map.items():
        print(f"  [{id_map[cid]}] {name}: {cls_count.get(id_map[cid], 0)}")
    print("总目标:", total_boxes)
    for s in SPLITS:
        nimg = sum(1 for a in assigns.values() if a == s)
        print(f"  {s}: {nimg} 图, 空标注 {empty.get(s, 0)}")

    # 写 data.yaml
    names = [name for cid, name in sorted(cat_map.items())]
    yaml_path = os.path.join(OUT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Blueberry maturity dataset (converted from COCO, Ni et al. 2020 Horticulture Research)\n")
        f.write("path: ../dataset\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: " + json.dumps(names) + "\n")
    print("data.yaml written:", yaml_path)


if __name__ == "__main__":
    main()
