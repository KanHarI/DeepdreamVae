import torch
from torchvision.models import ResNet  # type: ignore


class ResNetPartial(torch.nn.Module):
    def __init__(self, original_model: ResNet, n: int):
        super(ResNetPartial, self).__init__()
        self.features = torch.nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            *list(original_model.children())[
                4 : 4 + n
            ]  # Includes only the first N blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


def create_n_layers_resnet(n: int, original_resnet: ResNet) -> ResNetPartial:
    return ResNetPartial(original_resnet, n)
