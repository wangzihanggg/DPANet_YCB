import os

train_path = "/project/1_2301/DPANet-master/YCB_Video_Dataset/data"
"""
data
    -- 0000
        -- many rgb depth and mask
"""
sub_dirs = os.listdir(train_path)
rgb_paths = []
depth_paths = []
mask_paths = []
for sub_dir in sub_dirs:
    dir_path = os.path.join(train_path, sub_dir)
    all_files = os.listdir(dir_path)
    for file in all_files:
        if "color" in file:
            rgb_path = os.path.join(dir_path, file)
            depth_path = os.path.join(dir_path, file[:6] + "-depth.png")
            mask_path = os.path.join(dir_path, file[:6] + "-label.png")
            rgb_paths.append(rgb_path)
            depth_paths.append(depth_path)
            mask_paths.append(mask_path)

assert len(rgb_paths) == len(depth_paths) == len(mask_paths)
print(len(rgb_paths), len(depth_paths), len(mask_paths))


