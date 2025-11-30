import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    This is the fundamental building block of U-Net.
    It applies two convolution layers sequentially to extract features.
    """
    """Updated to handle mid-channels dynamically"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear interpolation for upsampling (lighter on memory)
        # or ConvTranspose2d (learnable upsampling)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding issues if input size is not perfectly divisible by 2
        # This ensures the tensor sizes match before concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Skip Connection: Concatenate along channel axis (dim=1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution to map features to classes"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
   

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER (Downsampling / Contracting Path) ---
        # The goal here is to capture Context (What is in the image?)
        # We increase the number of channels (features) while reducing image size.
        
        # Initial input: 6 channels -> 64 features
        self.inc = DoubleConv(n_channels, 64)
        
        # Down 1: 64 -> 128
        self.down1 = Down(64, 128)
        
        # Down 2: 128 -> 256
        self.down2 = Down(128, 256)
        
        # Down 3: 256 -> 512
        self.down3 = Down(256, 512)
        
        # Down 4: 512 -> 1024 (Bottleneck / Deepest point)
        # This layer holds the most abstract, high-level features.
        self.factor = 2 
        self.down4 = Down(512, 1024 // self.factor)

        # --- DECODER (Upsampling / Expanding Path) ---
        # The goal here is Localization (Where is it?).
        # We recover the image size using Up-Convolutions.
        
        # Up 1: 1024 -> 512 (Concatenates with Down 3 output)
        self.up1 = Up(1024, 512 // self.factor, self.factor)
        
        # Up 2: 512 -> 256
        self.up2 = Up(512, 256 // self.factor, self.factor)
        
        # Up 3: 256 -> 128
        self.up3 = Up(256, 128 // self.factor, self.factor)
        
        # Up 4: 128 -> 64
        self.up4 = Up(128, 64, self.factor)
        
        # --- OUTPUT LAYER ---
        # Reduces features to the number of classes (1 for binary mask)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Pass through the layers
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with Skip Connections
        # Notice we pass x4, x3, x2, x1 as second arguments.
        # These are the "Skip Connections" that carry spatial details.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    

if __name__ == "__main__":
    # Test the model with a dummy tensor
    # Batch Size: 1, Channels: 6, Height: 256, Width: 256
    x = torch.randn(1, 6, 256, 256)
    model = UNet(n_channels=6, n_classes=1)
    
    preds = model(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {preds.shape}")
    
    # Verify strict equality
    assert preds.shape == (1, 1, 256, 256)
    print("Network implementation successful!")