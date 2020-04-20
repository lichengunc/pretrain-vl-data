"""
This code makes COCO [train] captions, excluding the follows:
1) Karpathy's val and test images
2) RERER's val and test images
3) Flickr30K images

COCO [val] captions will be within Karpathy's [val] split.
"""
import os
import os.path as osp
import time
import json

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "..", "data")
coco_dir = osp.join(data_dir, "coco")
output_dir = osp.join(this_dir, "..", "output")

# load excluded_coco_vg_iids
Excluded = json.load(open(osp.join(output_dir,
                        "excluded_coco_vg_iids.json"), "r"))
excluded_coco_iids_set = set(Excluded["refer_val_coco_iids"] + \
                             Excluded["refer_test_coco_iids"] + \
                             Excluded["flickr30k_coco_iids"] + \
                             Excluded["karpathy_test_iids"])
val_coco_iids_set = set(Excluded["karpathy_val_iids"])
print(f"{len(excluded_coco_iids_set)} coco images will be excluded and "
      f"there are {len(val_coco_iids_set)} val coco images.")

# collect coco's captions
# - load coco_captions [{id, image_id, caption}]
coco_caption_data = json.load(open(osp.join(coco_dir, "annotations",
                            "captions_train2014.json")))["annotations"] + \
                    json.load(open(osp.join(coco_dir, "annotations",
                            "captions_val2014.json")))["annotations"]
train_caps, val_caps = [], []
for cap in coco_caption_data:
    if cap["image_id"] in excluded_coco_iids_set:
        continue
    data = {
        "coco_id": cap["image_id"],
        "caption": cap["caption"]
    }
    if cap["image_id"] in val_coco_iids_set:
        val_caps.append(data)
    else:
        train_caps.append(data)

# print stats
num_train_images = len(set([cap["coco_id"] for cap in train_caps]))
num_val_images = len(set([cap["coco_id"] for cap in val_caps]))
print(f"{len(train_caps)} [train] captions ({num_train_images} images) and "
      f"{len(val_caps)} [val] captions ({num_val_images} images) collected.")

# save
with open(osp.join(output_dir, "coco_train_captions.json"), "w") as f:
    json.dump(train_caps, f)
with open(osp.join(output_dir, "coco_val_captions.json"), "w") as f:
    json.dump(val_caps, f)
print(f"{osp.join(output_dir, 'coco_train_captions.json')} and "
      f"{osp.join(output_dir, 'coco_val_captions.json')} saved.")
