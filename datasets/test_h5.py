# import os
# import h5py
# import json  # 添加json模块导入

# window = 3
# all_windows = []
# file_path = "/datasets/graduate/datasets/mix-fluid-cconv/train/cd_0.0_cf_0.0.h5"
# with h5py.File(file_path, 'r') as h5f:
#     frames_group = h5f['frames']
#     frame_ids = sorted([int(k) for k in frames_group.keys()])
#     print(frames_group["1"].keys())
    
#     # 检查是否有足够的连续帧
#     if len(frame_ids) < window:
#         exit("Not enough frames in the file")
    
#     # 收集所有可能的窗口
#     for start_idx in range(len(frame_ids) - window + 1):
#         window_frames = frame_ids[start_idx:start_idx+window]
#         all_windows.append({
#             'file': file_path,
#             'frames': window_frames
#         })

# # all_windows to json
# # output_dir = os.path.dirname(file_path)
# # output_file = os.path.join(output_dir, "windows_data.json")
# output_file = "windows_data.json"

# # 将数据转为JSON格式
# # 注意：因为JSON不支持整数作为键，所以frames列表会被保留为数组
# json_data = {
#     "windows": all_windows,
#     "total_windows": len(all_windows),
#     "window_size": window
# }

# # 保存到文件
# with open(output_file, 'w') as f:
#     json.dump(json_data, f, indent=2)

# print(f"{len(all_windows)} windows")
# print(f"windows size: {window}")
# print(f"data to: {output_file}")

# # 打印第一个窗口示例
# if all_windows:
#     print("\first windows example:")
#     print(f"file: {all_windows[0]['file']}")
#     print(f"frames: {all_windows[0]['frames']}")

import numpy as np
a = np.float32([1000.0, 1000.0])  # 确保numpy可以正常使用
print(a)
print(type(a))