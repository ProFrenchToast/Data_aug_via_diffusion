import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.relu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(
                emb_dim,
                out_channels
            ),
            nn.ReLU()
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(
                emb_dim,
                out_channels
            ),
            nn.ReLU()
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down_large = Down(32, 64)
        #self.sa_down_large = SelfAttention(64, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 32)
        self.sa6 = SelfAttention(32, 64)
        self.up_large = Up(64, 32)
        #self.sa_up_large = SelfAttention(32, 128)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        #x_original = x
        #print(f"x {x.shape}")
        x0 = self.inc(x)
        #print(f"x0 {x0.shape}")
        x1 = self.down_large(x0, t)
        #print(f"x1 {x1.shape}")
        x2 = self.down1(x1, t)
        #print(f"x2 {x2.shape}")
        x2 = self.sa1(x2)
        #print(f"x2 {x2.shape}")
        x3 = self.down2(x2, t)
        #print(f"x3 {x3.shape}")
        x3 = self.sa2(x3)
        #print(f"x3 {x3.shape}")
        x4 = self.down3(x3, t)
        #print(f"x4 {x4.shape}")
        x4 = self.sa3(x4)
        #print(f"x4 {x4.shape}")
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        #print(f"x4 {x4.shape}, , x3 {x3.shape}")
        x = self.up1(x4, x3, t)
        #print(f"x {x.shape}")
        x = self.sa4(x)
        #print(f"x {x.shape}, x2 {x2.shape}")
        x = self.up2(x, x2, t)
        #print(f"x {x.shape}")
        x = self.sa5(x)
        #print(f"x {x.shape}, x1 {x1.shape}")
        x = self.up3(x, x1, t)
        #print(f"x {x.shape}")
        x = self.sa6(x)
        #print(f"x {x.shape}, x0 {x0.shape}")
        x = self.up_large(x, x0, t)
        #print(f"x {x.shape}")
        output = self.outc(x)
        #print(f"output {output.shape}")
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


import math
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


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
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, image_channels=3, 
                 down_channels = (64, 128, 256, 512, 1024),
                 up_channels = (1024, 512, 256, 128, 64),
                 time_emb_dim = 32):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
    
class simpleUnetAction(nn.Module):
    def __init__(self, action_size,
                 down_channels = (64, 128, 256, 512, 1024),
                 up_channels = (1024, 512, 256, 128, 64),
                 time_emb_dim = 256):
        super().__init__()
        self.Unet = SimpleUnet(image_channels=4,
                               down_channels=down_channels,
                               up_channels=up_channels,
                               time_emb_dim=time_emb_dim)
        self.action_size = action_size
        self.act_emb_layer = nn.Sequential(
            nn.Linear(
                action_size,
                128*128
            ),
            #nn.ReLU(),
        )
        self.act_decode_layer = nn.Sequential(
            nn.Linear(
                128*128,
                action_size
            ),
            #nn.ReLU()
        )

    def forward(self, state, action, timestep):
        action_emb = self.act_emb_layer(action)
        action_emb = action_emb.view(-1, 1, 128, 128)
        state_action = torch.cat((state, action_emb), dim=1)
        result = self.Unet(state_action, timestep)
        new_state = result[:, :3, :, :]
        new_action = self.act_decode_layer(result[:, 3, :, :].view(-1, 128*128))
        return new_state, new_action

class stateActionUnet(nn.Module):
    def __init__(self, action_size, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.Unet = UNet(c_in=4, c_out=4, time_dim=time_dim, device=device)
        self.action_size = action_size
        self.action_norm = torchvision.transforms.Compose([
            torchvision.transforms.Normalize((0.5), (0.5))])
        self.act_emb_layer = nn.Sequential(
            nn.Linear(
                action_size,
                128*128
            ),
            #nn.ReLU(),
        )
        self.act_decode_layer = nn.Sequential(
            nn.Linear(
                128*128,
                action_size
            ),
            #nn.ReLU()
        )

    def forward(self, state, action, timestep):
        action_emb = self.act_emb_layer(action)
        action_emb = action_emb.view(-1, 1, 128, 128)
        action_emb = self.action_norm(action_emb)
        state_action = torch.cat((state, action_emb), dim=1)
        result = self.Unet(state_action, timestep)
        new_state = result[:, :3, :, :]
        new_action = self.act_decode_layer(result[:, 3, :, :].view(-1, 128*128))
        return new_state, new_action


if __name__ == '__main__':
    # net = UNet(device="cpu")
    #net = UNet_conditional(num_classes=10, device="cpu")
    net = stateActionUnet(10).to("cuda")
    print(sum([p.numel() for p in net.parameters()]))
    x_s = torch.randn(3, 3, 128, 128).to("cuda")
    t = x_s.new_tensor([500] * x_s.shape[0]).float().to("cuda")
    x_a = torch.randn(3, 10).to("cuda")
    print(x_s.shape)
    print(x_a.shape)
    new_state, new_action = net(x_s, x_a, t)
    print(new_state.shape)
    print(new_action.shape)
