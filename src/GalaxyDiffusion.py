import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import h5py
import numpy as np
import math


class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = (q.transpose(-1, -2) @ k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        h_ = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(h_)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.silu(self.norm1(self.conv1(x)))
        h = h + self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return self.silu(h + self.shortcut(x))


class ScoreNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        
        self.inc = nn.Conv2d(3, channels, 3, padding=1)
        
        self.down1 = ResBlock(channels, channels * 2)
        self.down_conv = nn.Conv2d(channels * 2, channels * 2, 3, stride=2, padding=1)
        self.down2_block = ResBlock(channels * 2, channels * 4)
        
        self.mid1 = ResBlock(channels * 4, channels * 4)
        self.attn = AttentionBlock(channels * 4)
        self.mid2 = ResBlock(channels * 4, channels * 4)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_block = ResBlock(channels * 4 + channels * 2, channels * 2)
        self.up2 = ResBlock(channels * 2 + channels, channels)
        
        self.out = nn.Conv2d(channels, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        
        x3 = self.down_conv(x2)
        x3 = self.down2_block(x3, t_emb)
        
        x4 = self.mid1(x3, t_emb)
        x4 = self.attn(x4)
        x4 = self.mid2(x4, t_emb)
        
        x5 = self.upsample(x4)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.up1_block(x5, t_emb)
        
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.up2(x6, t_emb)
        
        return self.out(x6)

class Diffusion:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.steps = steps
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, 64, 64)).to(next(model.parameters()).device)
            for i in tqdm(reversed(range(0, self.steps)), position=0):
                t = (torch.ones(n) * i).long().to(x.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

class Galaxy10Dataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['ans'])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = Galaxy10Dataset("/kaggle/input/datasets/tiborcornelli/galaxy10/Galaxy10.h5")
    indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=True)
    
    model = ScoreNet().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    ema = EMA(model.module if isinstance(model, nn.DataParallel) else model)
    optimizer = Adam(model.parameters(), lr=1e-4)
    diffusion = Diffusion(device=device)
    
    EPOCHS = 100
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        pbar.set_description(f"Epoch {epoch} / {EPOCHS} ")
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = torch.randint(low=0, high=1000, size=(images.shape[0],)).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            
            predicted_noise = model(x_t, t)
            loss = nn.functional.mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            pbar.set_postfix(MSE=loss.item())

        if (epoch + 1) % 10 == 0:
            ema.apply_shadow()
            sampling_model = model.module if isinstance(model, nn.DataParallel) else model
            samples = diffusion.sample(sampling_model, 16)
            samples = (samples.clamp(-1, 1) + 1) / 2
            utils.save_image(samples, f"samples_epoch_{epoch+1}.png", nrow=4)
            ema.restore()

    ema.apply_shadow()
    final_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_model_state, "galaxy_diffusion_model.pth")

if __name__ == "__main__":
    train()