"""
We collect pretrained V+L data from sbucaptions after excluding:
1) Karpathy's val+test images
2) refer's val+test images
3) all Flickr30K images

The excluded_flickr_url_ids were pre-computed by running
```python prepro/get_excluded_iids.py```
"""
import os
import os.path as osp
import json
from tqdm import tqdm

# paths
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, "..", "data")
sbu_dir = osp.join(data_dir, "sbucaptions")
output_dir = osp.join(this_dir, "..", "output")

# load excluded_flickr_url_ids
Excluded = json.load(open(osp.join(output_dir,
                        "excluded_coco_vg_iids.json"), "r"))
excluded_flickr_url_ids = Excluded["excluded_flickr_url_ids"]
excluded_flickr_url_ids_set = set(excluded_flickr_url_ids)

# load sbucaptions data (UNC version)
# TODO: change Line 32-38 to be the format of the data downloaded
# from http://www.cs.virginia.edu/~vicente/sbucaptions/
data = []
cnt = 0
for line in tqdm(open(osp.join(sbu_dir, "vicente-flickr-1M-list-v2.txt"),
                "r").readlines()):
    items = str(line).split('\t')
    assert len(items) == 4, items
    # example of sbu_id: 4385058960_b0f291553e_2723_4385058960
    fpath, token, sbu_id, sent = items
    url_id = int(sbu_id.split('_')[-1])
    if url_id in excluded_flickr_url_ids_set:
        cnt += 1
        print(f"url_id[{url_id}] is forbidden, {cnt} of such have been found.")
    else:
        data.append({
            "file_path": fpath,
            "sbu_id": sbu_id,
            "caption": sent.strip()
        })

# save
print(f"In total, {cnt} images in sbucaptions dataset were forbidden.")
with open(osp.join(output_dir, "sbu_captions.json"), "w") as f:
    json.dump(data, f)
print("output/sbu_captions.json saved.")

# TODO:
# 1) download and check alive images from the left urls
# 2) randomly split out 10K images to be [val] split during pre-training
