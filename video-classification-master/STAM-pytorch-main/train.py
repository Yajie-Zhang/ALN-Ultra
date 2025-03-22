import torch

from models.stam import STAM

model = STAM(
    dim = 512,
    image_size = (256,400),     # size of image
    patch_size = (16,20),      # patch size
    num_frames = 100,       # number of image frames, selected out of video
    space_depth = 12,     # depth of vision transformer
    space_heads = 8,      # heads of vision transformer
    space_mlp_dim = 2048, # feedforward hidden dimension of vision transformer
    time_depth = 6,       # depth of time transformer (in paper, it was shallower, 6)
    time_heads = 8,       # heads of time transformer
    time_mlp_dim = 2048,  # feedforward hidden dimension of time transformer
    num_classes = 100,    # number of output classes
    space_dim_head = 64,  # space transformer head dimension
    time_dim_head = 64,   # time transformer head dimension
    dropout = 0.,         # dropout
    emb_dropout = 0.      # embedding dropout
)

frames = torch.randn(2, 100, 3, 256,400) # (batch x frames x channels x height x width)
pred = model(frames) # (2, 100)
print(pred.shape)