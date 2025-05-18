import torch

foo = torch.tensor([1,2,3])
foo = foo.to('cuda')

props = torch.cuda.get_device_properties(0)
print(props)