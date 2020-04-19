"""
We will get to-be-excluded images' ids by checking:
1) Karpathy's test split
2) refcoco/refcoco+/refcocog's val+test split
3) duplicated Flickr30k images in COCO
Note, karpathy's val split will be our val split for pre-training.
"""
import os
import os.path as osp
import json
import pickle

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "../data")
vg_dir = osp.join(data_dir, "vg")
coco_dir = osp.join(data_dir, "coco")
refer_dir = osp.join(data_dir, "refer")
flickr_dir = osp.join(data_dir, "flickr30k")
karpathy_splits_dir = osp.join(coco_dir, "karpathy_splits")
output_dir = osp.join(this_dir, "../output")

# exclude refcoco/refcoco+/refcocog's val+test images
refcoco_data = pickle.load(open(osp.join(refer_dir,
                        "refcoco/refs(unc).p"), "rb")) # same as refcoco+
refcocog_data = pickle.load(open(osp.join(refer_dir,
                        "refcocog/refs(umd).p"), "rb"))
refer_val_coco_iids = []
refer_test_coco_iids = []
for ref in refcoco_data:
    if ref["split"] in ["testA", "testB"]:
        refer_test_coco_iids.append(ref["image_id"])
    if ref["split"] == "val":
        refer_val_coco_iids.append(ref["image_id"])
for ref in refcocog_data:
    if ref["split"] in ["test"]:
        refer_test_coco_iids.append(ref["image_id"])
    if ref["split"] == "val":
        refer_val_coco_iids.append(ref["image_id"])
refer_val_coco_iids_set = set(refer_val_coco_iids)
refer_test_coco_iids_set = set(refer_test_coco_iids)
print(f"In refcoco/refcoco+/refcocog, there are "
      f"{len(refer_val_coco_iids_set)} [val] images and "
      f"{len(refer_test_coco_iids_set)} [test] images in COCO's [train] split.")

# load Karpathy's splits
karpathy_train_iids = []
karpathy_train_file = open(osp.join(karpathy_splits_dir,
                                "karpathy_train_images.txt"), "r")
for x in karpathy_train_file.readlines():
    karpathy_train_iids.append(int(x.split()[1]))
assert len(set(karpathy_train_iids)) == len(karpathy_train_iids)
print(f"COCO\'s [karpathy_train] has {len(karpathy_train_iids)} images.")

karpathy_val_iids = []
karpathy_val_file = open(osp.join(karpathy_splits_dir,
                                  "karpathy_val_images.txt"), "r")
for x in karpathy_val_file.readlines():
    karpathy_val_iids.append(int(x.split()[1]))
assert len(set(karpathy_val_iids)) == len(karpathy_val_iids)
print(f"COCO\'s [karpathy_val] has {len(karpathy_val_iids)} images.")

karpathy_test_iids = []
karpathy_test_file = open(osp.join(karpathy_splits_dir,
                                   "karpathy_test_images.txt"), "r")
for x in karpathy_test_file.readlines():
    karpathy_test_iids.append(int(x.split()[1]))
assert len(set(karpathy_test_iids)) == len(karpathy_test_iids)
print(f"COCO\'s [karpathy_test] has {len(karpathy_test_iids)} images.")

# exclude all Flickr30K images from COCO and VG for zero-shot retrieval
# coco session
flickr30k_coco_iids = []
flickr30k_vg_iids = []
flickr30k_url_ids = set()
for url_id in open(osp.join(flickr_dir,
                   "flickr30k_entities", "train.txt"), "r").readlines():
  flickr30k_url_ids.add(int(url_id))
for url_id in open(osp.join(flickr_dir,
                   "flickr30k_entities", "val.txt"), "r").readlines():
  flickr30k_url_ids.add(int(url_id))
for url_id in open(osp.join(flickr_dir,
                   "flickr30k_entities", "test.txt"), "r").readlines():
  flickr30k_url_ids.add(int(url_id))
print(f"There are {len(flickr30k_url_ids)} flickr30k_url_ids.")

coco_image_data = json.load(open(osp.join(coco_dir, "annotations",
                            "instances_train2014.json")))["images"] + \
                  json.load(open(osp.join(coco_dir, "annotations",
                            "instances_val2014.json")))["images"]
for img in coco_image_data:
    # example: 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg'
    url_id = int(img["flickr_url"].split("/")[-1].split("_")[0])
    if url_id in flickr30k_url_ids:
        flickr30k_coco_iids.append(img["id"])
print(f"{len(flickr30k_coco_iids)} coco images were found in Flickr30K.")

# vg session
vg_image_data = json.load(open(osp.join(vg_dir, "image_data.json")))
for img in vg_image_data:
  if img["flickr_id"] is not None:
    url_id = int(img["flickr_id"])
    if url_id in flickr30k_url_ids:
      flickr30k_vg_iids.append(img["image_id"])
print(f"{len(flickr30k_vg_iids)} vg images were found in Flickr30K.")

# # some random testing output
# print(len(refer_val_coco_iids_set.intersection(refer_test_coco_iids_set)))
# print(len(set(karpathy_val_iids).intersection(set(karpathy_test_iids))))

# Save
output = {"refer_val_coco_iids": list(refer_val_coco_iids_set),
          "refer_test_coco_iids": list(refer_test_coco_iids_set),
          "flickr30k_coco_iids": flickr30k_coco_iids,
          "flickr30k_vg_iids": flickr30k_vg_iids,
          "karpathy_train_iids": karpathy_train_iids,
          "karpathy_val_iids": karpathy_val_iids,
          "karpathy_test_iids": karpathy_test_iids}
with open(f"{output_dir}/excluded_coco_vg_iids.json", "w") as f:
    json.dump(output, f)
print("output/excluded_coco_vg_iids.json saved.")
