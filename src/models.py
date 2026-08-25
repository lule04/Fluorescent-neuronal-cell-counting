import torch
import torch.nn as nn


def _activation(name):
    if name == "elu":
        return nn.ELU(inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Nepoznata aktivacija: {name}")


# ---------------------------------------------------------------------------
# Rezidualni blok (BN → Act → Conv3x3 → BN → Act → Conv3x3 + shortcut)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="elu"):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            _activation(activation),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            _activation(activation),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv_branch(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Attention Gate (Oktay et al. 2018)
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ---------------------------------------------------------------------------
# c-ResUnet  (Morelli & Clissa 2021,  kanali 16→32→64→128)
# ---------------------------------------------------------------------------

class CResUnet(nn.Module):
    def __init__(self, in_channels=3, activation="elu", use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        act = activation

        # --- encoder ---
        self.init_conv = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        self.enc1 = ResBlock(16, 16, act)
        # dodatni 5x5 konv blok na prvom nivou (c-ResUnet specifikacija, Fig. 1)
        self.enc1_5x5 = nn.Sequential(
            nn.BatchNorm2d(16),
            _activation(act),
            nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = ResBlock(16, 32, act)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = ResBlock(32, 64, act)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # --- bottleneck ---
        self.bottleneck = ResBlock(64, 128, act)

        # --- decoder ---
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.dec3 = ResBlock(128, 64, act)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)
        self.dec2 = ResBlock(64, 32, act)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False)
        self.dec1 = ResBlock(32, 16, act)

        # --- izlaz ---
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # --- attention gates (opciono) ---
        if use_attention:
            self.ag3 = AttentionGate(F_g=64, F_l=64, F_int=32)
            self.ag2 = AttentionGate(F_g=32, F_l=32, F_int=16)
            self.ag1 = AttentionGate(F_g=16, F_l=16, F_int=8)

    def forward(self, x):
        # encoder
        x0 = self.init_conv(x)

        e1 = self.enc1(x0)
        e1 = e1 + self.enc1_5x5(e1)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck(p3)

        # decoder
        d3 = self.up3(b)
        skip3 = self.ag3(g=d3, x=e3) if self.use_attention else e3
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        skip2 = self.ag2(g=d2, x=e2) if self.use_attention else e2
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        skip1 = self.ag1(g=d1, x=e1) if self.use_attention else e1
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Fabrika + inicijalizacija
# ---------------------------------------------------------------------------

def build_model(attention=False, activation="elu", in_channels=3):
    model = CResUnet(
        in_channels=in_channels,
        activation=activation,
        use_attention=attention,
    )
    model.apply(init_weights)
    return model


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
