import torch
import torch.nn as nn
import torchvision.models as models

# ─── models ──────────────────────────────────────────────────────────────────


class DeepElementwiseModel(nn.Module):
    def forward(self, x):
        for _ in range(40):
            x = torch.sigmoid(torch.relu(x + 1))
        return x.mean()


class EnhancedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2
        )
        self.pattern_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        for _ in range(30):
            a = x
            b = self.pattern_conv(x)
            c = self.pattern_conv(x)
            x = a*c + a*b
            x = torch.sigmoid(torch.relu(x + 1))
        return self.fc(x.mean(dim=[2, 3]))


class EnhancedVGG(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.vgg16(weights=None)
        self.features = base.features
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        # a short elementwise chain
        x = torch.tanh(torch.exp(x + 2))
        x = torch.sigmoid(x)
        x = torch.sqrt(x + 1)
        return self.fc(x.mean(dim=[2, 3]))


class BitShiftTestModel(nn.Module):
    def forward(self, x: torch.Tensor):
        shifted = torch.bitwise_left_shift(x, 1)
        return torch.sum(shifted)


class RewriteTriggerResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(
            *list(backbone.children())[:-1])  # Remove FC
        self.head = nn.Linear(512, 10)

    def forward(self, image, B, C):
        # === Real path (ResNet on image input) ===
        x = self.features(image)
        x = torch.flatten(x, 1)

        # === Rewrite patterns on vector-shaped inputs ===
        BIG_B = B.repeat(1, 64)
        BIG_C = C.repeat(1, 64)

        for _ in range(100):  # Repeat 100x to amplify FLOP impact

            # 1. prod(exp(x)) → exp(sum(x))
            _ = torch.prod(torch.exp(BIG_B), dim=1)

            # 2. (B * sqrt(C)) * (sqrt(C) * B)
            s = torch.sqrt(BIG_C)
            _ = (BIG_B * s) * (s * BIG_B)

            # 3. sum(bitshift(x)) → bitshift(sum(x))
            shifted = torch.bitwise_left_shift(BIG_B.to(torch.int32), 2)
            summed_int = torch.sum(shifted, dim=1, keepdim=True)
            _ = summed_int.float()

            # 4. A * B + A * C → A * (B + C)
            _ = BIG_B * BIG_C + BIG_B * BIG_B

            # 5. A + A * B → A * (B + 1)
            _ = BIG_B + BIG_B * BIG_C

            # 6. (1/B) * (1/(B*C)) → (1/B)^2 * (1/C)
            _ = torch.reciprocal(BIG_B) * torch.reciprocal(BIG_B * BIG_C)

        return self.head(x)


class RewriteTriggerResNetWrapped(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        A = x[:, :, :, :224]  # [32,3,224,224]
        B = x[:, 0, 0, 224:224+8192]    # [32,8192]
        C = x[:, 0, 0, 224+8192:224+16384]  # [32,8192]
        return self.base_model(A, B, C)


class ComprehensiveRewriteModel(nn.Module):
    """
    A single‑module harness containing all of the algebraic rewrites:
      1) Bitshift→Sum
      2) Prod(Exp)→Exp(Sum)
      3) Recip‑Associative
      4) Sqrt‑Associative
      5) Distributive A*B + A*C → A*(B+C)
      6) Distributive2 A + A*B → A*(1+B)
    """

    def forward(self, x):
        # Build two helper tensors B and C from x
        B = x + 1.0
        C = x + 2.0

        # 1) Bitshift → Sum
        x_int = x.to(torch.int32)
        t1 = torch.bitwise_left_shift(x_int, 1)
        sum_int = torch.sum(t1, dim=1, keepdim=True)
        p1 = sum_int.float()

        # 2) Prod(Exp) → Exp(Sum)
        t2 = torch.exp(B)
        p2 = torch.prod(t2, dim=1, keepdim=True)

        # 3) Recip‑Associative: Recip(A) * Recip(A * B) → (Recip(A))² * Recip(B)
        rA = torch.reciprocal(B)
        rAB = torch.reciprocal(B * C)
        p3 = rA * rAB

        # 4) Sqrt‑Associative: (A*√B)*(√B*C) → A*B*C
        sB = torch.sqrt(B)
        m1 = x * sB
        m2 = sB * C
        p4 = m1 * m2

        # 5) Distributive: A*B + A*C → A*(B+C)
        d5 = x * B + x * C

        # 6) Distributive2: A + A*B → A*(1 + B)
        d6 = x + x * C

        # Combine everything and collapse to a scalar
        out = p1 + p2 + p3 + p4 + d5 + d6
        return out.mean()
