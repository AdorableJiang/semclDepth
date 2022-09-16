"""
This script generate `cityscapes_test.txt` with the same style of `splits/cityscapes_train.txt` and `splits/cityscapes_val.txt`.
Before run this script, make sure cityscapes dataset is all set as described in `docs/dataset_prepare.md`.
"""

import os
import os.path as osp
import warnings

data_root = 'data/cityscapes'

leftimg8bit_root=osp.join(data_root,'leftImg8bit','test')
disparity_root=osp.join(data_root,'disparity','test')
camera_root=osp.join(data_root,'camera','test')

# check existence of three paths
assert osp.exists(leftimg8bit_root), f"{leftimg8bit_root} does not exists."
assert osp.exists(disparity_root), f"{disparity_root} does not exists."
assert osp.exists(camera_root), f"{camera_root} does not exists."

leftimg8bit_imgs=[]
for one_dir in os.listdir(leftimg8bit_root):#,topdown=False
    for one_file in os.listdir(osp.join(leftimg8bit_root,one_dir)):
        leftimg8bit_imgs.append(osp.join(leftimg8bit_root[len('data/cityscapes/'):],one_dir,one_file))
print(len(leftimg8bit_imgs))

assert len(leftimg8bit_imgs)==1525, "Wrong amount of samples. There should be 1525 test samples in total but got {len(leftimg8bit_imgs)}." # See https://www.cityscapes-dataset.com/downloads/.
# print(leftimg8bit_imgs)

disparity_imgs=[]
camera_jsons=[]
for one_test_img in leftimg8bit_imgs:
    stem=one_test_img[len('leftImg8bit/'):-(len('leftImg8bit.png'))]
    # disparity
    one_disparity='disparity/'+stem+'disparity.png'
    assert osp.exists(osp.join(data_root,one_disparity)),f"Disparity file '{one_disparity}' does not exist."
    disparity_imgs.append(one_disparity)
    # camera json
    one_camera=stem+'camera.json'
    assert osp.exists(osp.join(data_root,one_disparity)),f"Camera json file '{one_disparity}' does not exist."
    camera_jsons.append(one_camera)

print(len(disparity_imgs))
print(len(camera_jsons))

# write to scityscapes_test.txt
cityscapes_test_txt='splits/cityscapes_test.txt'
if osp.exists(cityscapes_test_txt): 
    warnings.warn(f"The existing {cityscapes_test_txt} will be deleted before generating a new one.")
    os.remove(cityscapes_test_txt)

with open(cityscapes_test_txt, 'w') as f:
    for i in range(len(leftimg8bit_imgs)):
        f.write(leftimg8bit_imgs[i]+' '+disparity_imgs[i]+' '+camera_jsons[i]+'\n')
f.close()
print(f"Please copy {cityscapes_test_txt} to your cityscapes path just as you did to other split files.")