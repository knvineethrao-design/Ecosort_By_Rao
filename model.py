import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            DepthwiseSeparableConv(16, 24, stride=2),
            DepthwiseSeparableConv(24, 32, stride=2),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 64, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == "__main__":
    model = WasteClassifier()

    sample = torch.randn(1, 3, 96, 96)
    output = model(sample)

    total_params = sum(p.numel() for p in model.parameters())

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Number of parameters:", total_params)
    print("Output:", output)

    assert output.shape == (1, 6)
    assert total_params < 350_000

    print("Model architecture test passed.")