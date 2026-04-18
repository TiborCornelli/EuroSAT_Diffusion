import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_eurosat_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    highway_idx = full_dataset.class_to_idx['AnnualCrop']
    indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == highway_idx]
    highway_subset = torch.utils.data.Subset(full_dataset, indices)
    return DataLoader(highway_subset, batch_size=batch_size, shuffle=True)

def normalize_eurosat(x, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

def denormalize_eurosat(x, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
    x = x * std + mean
    return torch.clamp(x, 0, 1)

def forward_diffusion(x_0, t):
    t = t.view(-1, 1, 1, 1)
    mean_coef = torch.exp(-t)
    var_coef = 1 - torch.exp(-2 * t)
    std = torch.sqrt(var_coef)
    epsilon = torch.randn_like(x_0)
    return mean_coef * x_0 + std * epsilon, epsilon

class ResBlock(nn.Module):
    def __init__(self, channels, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels)
        )
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.silu = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        # Inject time embedding
        h = h + self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.res = ResBlock(in_channels, emb_dim)
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.down(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res = ResBlock(out_channels, emb_dim)

    def forward(self, x, t_emb):
        x = self.up(x)
        x = self.res(x, t_emb)
        return x

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        self.inc = nn.Conv2d(3, 64, 3, padding=1)
        
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        
        self.mid = ResBlock(512)
        
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t.view(-1, 1))
        
        h = self.inc(x)
        h = self.down1(h, t_emb)
        h = self.down2(h, t_emb)
        h = self.down3(h, t_emb)
        
        h = self.mid(h, t_emb)
        
        h = self.up1(h, t_emb)
        h = self.up2(h, t_emb)
        h = self.up3(h, t_emb)
        
        return self.out(h)

def denoising_score_loss(score_net, x_0, t):
    x_t, epsilon = forward_diffusion(x_0, t)
    predicted_noise = score_net(x_t, t)
    return nn.functional.mse_loss(predicted_noise, epsilon)

def sample_ula(score_net, shape, steps, h):
    device = next(score_net.parameters()).device
    x = torch.randn(shape, device=device)
    score_net.eval()
    with torch.no_grad():
        for k in range(steps, 0, -1):
            t_val = k / steps
            t = torch.full((shape[0],), t_val, device=device)
            xi = torch.randn_like(x)
            score = score_net(x, t)
            x = x + h * score + torch.sqrt(torch.tensor(2 * h)) * xi
    return x

def run_training(score_net, dataloader, mean, std, epochs=100, lr=5e-5):
    optimizer = Adam(score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = next(score_net.parameters()).device
    score_net.train()
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x_raw = batch[0].to(device)
            x_0 = normalize_eurosat(x_raw, mean, std)
            t = torch.rand((x_0.shape[0],), device=device)
            
            optimizer.zero_grad()
            loss = denoising_score_loss(score_net, x_0, t)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            
        scheduler.step()
        history.append(epoch_loss / len(dataloader))
    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    data_path = "./Data"
    
    temp_loader = get_eurosat_loader(data_path, batch_size=32)
    
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    for data, _ in temp_loader:
        sum_ += torch.mean(data, dim=[0, 2, 3]) * data.size(0)
        sum_sq += torch.mean(data**2, dim=[0, 2, 3]) * data.size(0)
        count += data.size(0)
    
    mean = (sum_ / count).tolist()
    std = torch.sqrt((sum_sq / count) - (sum_ / count)**2).tolist()
    
    model = ScoreNet().to(device)
    
    losses = run_training(
        score_net=model,
        dataloader=temp_loader,
        mean=mean,
        std=std,
        epochs=100,
        lr=1e-4
    )

    torch.save(model.state_dict(), "score_net_eurosat.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss_history.png')
    
    raw_samples = sample_ula(model, (16, 3, 64, 64), steps=1000, h=0.01)
    samples = denormalize_eurosat(raw_samples, mean, std)
    
    grid = utils.make_grid(samples, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Generated Highway Samples")
    plt.axis("off")
    plt.savefig('examples.png')