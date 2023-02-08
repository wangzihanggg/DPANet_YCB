# 在高版本的pytorch运行才行
import torch
state_dict = torch.load("/project/1_2301/DPANet-master/res50_res/model-1")
torch.save(state_dict, "new_model.pth", _use_new_zipfile_serialization=False)