"""
This code makes vg_questions and vg_annotations, excluding the follows:
1) Karpathy's val images
We don't bother excluding REFER's val and test images as this dataset will only
be used for finetuning, and the vqa validation will be conducted on 
1) Karpathy's val
2) COCO's test-dev

The original vg_questions and vg_annotations are downloaded from MCAN:
https://github.com/MILVLG/mcan-vqa
"""
import os
import os.path as osp
import time
import json

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "..", "data")
vqa_dir = osp.join(data_dir, "vqa")
output_dir = osp.join(this_dir, "..", "output")

# load excluded_coco_vg_iids
Excluded = json.load(open(osp.join(output_dir,
                        "excluded_coco_vg_iids.json"), "r"))
karpathy_train_iids = set(Excluded["karpathy_train_iids"])
karpathy_val_iids = set(Excluded["karpathy_val_iids"])
karpathy_test_iids = set(Excluded["karpathy_test_iids"])

# collect vg's questions
vg_questions = json.load(open(osp.join(vqa_dir, 
                            "vg_questions.json")))["questions"]
vg_annotations = json.load(open(osp.join(vqa_dir, 
                            "vg_annotations.json")))["annotations"]

ktrain_questions, ktrain_annotations = [], []
kval_questions, kval_annotations = [], []
ktest_questions, ktest_annotations = [], []
for q, a in zip(vg_questions, vg_annotations):
    assert q["image_id"] == a["image_id"]
    if q["image_id"] in karpathy_train_iids:
        ktrain_questions.append(q)
        ktrain_annotations.append(a)
    elif q["image_id"] in karpathy_val_iids:
        kval_questions.append(q)
        kval_annotations.append(a)
    else:
        assert q["image_id"] in karpathy_test_iids
        ktest_questions.append(q)
        ktest_annotations.append(a)

# print stats
num_ktrain_images = len(set([q["image_id"] for q in ktrain_questions]))
num_kval_images = len(set([q["image_id"] for q in kval_questions]))
num_ktest_images = len(set([q["image_id"] for q in ktest_questions]))
print(f"{len(ktrain_questions)} [ktrain] questions ({num_ktrain_images} images)"
      f", {len(kval_questions)} [kval] questions ({num_kval_images} images)"
      f", {len(ktest_questions)} [ktest] questions ({num_ktest_images} images) "
      f"collected.")

# save
Questions = {
    "ktrain": ktrain_questions,
    "kval": kval_questions,
    "ktest": ktest_questions
}
Annotations = {
    "ktrain": ktrain_annotations,
    "kval": kval_annotations,
    "ktest": ktest_annotations
}
for split in ["ktrain", "kval", "ktest"]:
    with open(osp.join(output_dir, f"vg_{split}_questions.json"), "w") as f:
        json.dump({"questions": Questions[split]}, f)
    with open(osp.join(output_dir, f"vg_{split}_annotations.json"), "w") as f:
        json.dump({"annotations": Annotations[split]}, f)
    print(f"vg_{split}_questions.json and vg_{split}_annotations.json saved in"
          f" {output_dir}.")