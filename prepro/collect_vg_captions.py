"""
This code makes VG [train] captions, excluding the follows:
1) Karpathy's val and test images
2) RERER's val and test images
3) Flickr30K images

VG [val] captions will be within Karpathy's [val] split.
"""
import os
import os.path as osp
import time
import json

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "..", "data")
vg_dir = osp.join(data_dir, "vg")
output_dir = osp.join(this_dir, "..", "output")

# load vg's image_data = [{image_id, coco_id, height, width, flickr_id, url}]
vg_image_data = json.load(open(osp.join(vg_dir, "image_data.json")))
vg_iid_to_img = {img["image_id"]: img for img in vg_image_data}

# load excluded_coco_vg_iids
Excluded = json.load(open(osp.join(output_dir,
                        "excluded_coco_vg_iids.json"), "r"))
excluded_coco_iids_set = set(Excluded["refer_val_coco_iids"] + \
                             Excluded["refer_test_coco_iids"] + \
                             Excluded["flickr30k_coco_iids"] + \
                             Excluded["karpathy_test_iids"])
val_coco_iids_set = set(Excluded["karpathy_val_iids"])

# excluded_vg_iids
excluded_vg_iids_set = set()
excluded_vg_iids_set |= set(Excluded["flickr30k_vg_iids"])  # no flickr30k
for img in vg_image_data:
    if img["coco_id"] is not None and img["coco_id"] in excluded_coco_iids_set:
        excluded_vg_iids_set.add(img["image_id"])
print(f"{len(excluded_vg_iids_set)} vg images will be excluded.")

# collect VG captions
# - load vg_captions [{id, regions}]
# - where regions = [{region_id, image_id, phrase, x, y, width, height}]
vg_caption_data = json.load(open(osp.join(vg_dir, "region_descriptions.json")))
train_caps, val_caps = [], []
for item in vg_caption_data:
    # {image_id, coco_id, flickr_id, height, width, url}
    vg_img = vg_iid_to_img[item["id"]]
    if vg_img["image_id"] in excluded_vg_iids_set:
        continue
    # add to collections
    for reg in item["regions"]:
        # reg = {x, y, width, height, phrase, region_id, image_id}
        if len(reg["phrase"].split()) == 0:
            print("void densecap found.")
            continue  # 7 dense caps are void.
        data = {
            "vg_id": vg_img["image_id"],
            "coco_id": vg_img["coco_id"],  # added for reference, could be None
            "caption" : reg["phrase"],
            "bbox" : [reg['x'], reg['y'], reg['width'], reg['height']],
        }
        if vg_img["coco_id"] and vg_img["coco_id"] in val_coco_iids_set:
            val_caps.append(data)
        else:
            train_caps.append(data)

# print stats
num_train_images = len(set([cap["vg_id"] for cap in train_caps]))
num_val_images = len(set([cap["vg_id"] for cap in val_caps]))
print(f"{len(train_caps)} [train] captions ({num_train_images} images) and "
      f"{len(val_caps)} [val] captions ({num_val_images} images) collected.")

# save
with open(osp.join(output_dir, "vg_train_captions.json"), "w") as f:
    json.dump(train_caps, f)
with open(osp.join(output_dir, "vg_val_captions.json"), "w") as f:
    json.dump(val_caps, f)
print(f"{osp.join(output_dir, 'vg_train_captions.json')} and "
      f"{osp.join(output_dir, 'vg_val_captions.json')} saved.")
