import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, H', W')
        x = x.flatten(2)  # Shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # Residual connection
        x = x + self.ffn(self.ln2(x))        # Residual connection
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, embed_dim, num_heads, num_layers, hidden_dim, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, num_patches + 1, embed_dim)
        # x += self.positional_embedding  # Adding positional embedding

        for block in self.transformer_blocks:
            x = block(x)

        cls_output = x[:, 0]  # 取出 CLS token 的输出
        return self.fc_out(cls_output)  # 输出分类结果

def create_vit_base(img_size=(256, 400), num_classes=1000):
    # ViT-base hyperparameters
    patch_size = 16
    embed_dim = 768  # embedding dimension
    num_heads = 12  # number of attention heads
    num_layers = 12  # number of transformer layers
    hidden_dim = 3072  # hidden dimension of feedforward layer

    # 计算输入图像的通道数，通常RGB图像的通道数是3
    in_channels = 3
    model = VisionTransformer(
        img_size=img_size[0],  # 输入图像高度
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    return model

# # 示例：创建一个 Vision Transformer 模型
# model = create_vit_base(img_size=(256, 400), num_classes=1000)
#
# # 测试模型的输出
# dummy_input = torch.randn(1, 3, 256, 400)  # batch size=1, channels=3, height=256, width=400
# output = model(dummy_input)
# print(output.shape)  # 应该是 (1, 1000)，对应于 num_classes
