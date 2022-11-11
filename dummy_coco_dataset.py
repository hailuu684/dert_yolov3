from .ultis.import_packages import *

"""
Sample label's data
[{'boxes': tensor([[0.5517, 0.5152, 0.0189, 0.0103], [0.7169, 0.5373, 0.0136, 0.0475]]), 'labels': tensor([1, 3], 
dtype=torch.int32), 'area': tensor([ 72.1889, 240.6667]), 'iscrowd': tensor([0, 0], dtype=torch.int32), 'image_id': 
tensor([5]), 'orig_size': tensor([[1024, 1024]], dtype=torch.int32), 'size': tensor([611, 608])}, 

{'boxes': tensor([[
0.2182, 0.6104, 0.0273, 0.0144], [0.5685, 0.6099, 0.0269, 0.0139]]), 'labels': tensor([1, 1], dtype=torch.int32), 
'area': tensor([263.1899, 251.6712]), 'iscrowd': tensor([0, 0], dtype=torch.int32), 'image_id': tensor([6]), 
'orig_size': tensor([[1024, 1024]], dtype=torch.int32), 'size': tensor([841, 800])}, 

{'boxes': tensor([[0.3852, 
0.5123, 0.0195, 0.0103], [0.4702, 0.5156, 0.0054, 0.0222]]), 'labels': tensor([1, 3], dtype=torch.int32), 
'area': tensor([129.7582,  76.2827]), 'iscrowd': tensor([0, 0], dtype=torch.int32), 'image_id': tensor([7]), 
'orig_size': tensor([[1024, 1024]], dtype=torch.int32), 'size': tensor([802, 800])}, 

{'boxes': tensor([[0.6307, 
0.5169, 0.0205, 0.0108], [0.3040, 0.5205, 0.0244, 0.0131], [0.6361, 0.5165, 0.0157, 0.0138], [0.2886, 0.5218, 0.0207, 
0.0150], [0.5211, 0.5213, 0.0057, 0.0243]]), 'labels': tensor([1, 1, 2, 2, 3], dtype=torch.int32), 'area': tensor([
65.7481, 94.3653, 64.3010, 92.1203, 40.8475]), 'iscrowd': tensor([0, 0, 0, 0, 0], dtype=torch.int32), 'image_id': 
tensor([8]), 'orig_size': tensor([[1024, 1024]], dtype=torch.int32), 'size': tensor([544, 544])}, 

{'boxes': tensor([[
0.3576, 0.5184, 0.0225, 0.0118], [0.7169, 0.5227, 0.0274, 0.0145], [0.3510, 0.5180, 0.0173, 0.0151], [0.5018, 0.5244, 
0.0063, 0.0279]]), 'labels': tensor([1, 1, 2, 3], dtype=torch.int32), 'area': tensor([ 69.5155, 104.4127,  68.4392,  
46.1463]), 'iscrowd': tensor([0, 0, 0, 0], dtype=torch.int32), 'image_id': tensor([9]), 'orig_size': tensor([[1024, 
1024]], dtype=torch.int32), 'size': tensor([512, 512])}] 

Sample image's data
size(3,any,any) - normalized
"""