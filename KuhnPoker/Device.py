import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU :-)")
    device = torch.device("cuda:0")
else:
    print("Using CPU :-(")
    device = torch.device("cpu")


