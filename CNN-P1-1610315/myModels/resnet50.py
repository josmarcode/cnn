import torch
from torch import nn
import matplotlib.pyplot as plt


class RESTNET50(nn.Module):
    """
    Neural Network based on RESTNET50 architecture
    to classify images from CIFAR-100 dataset.

    Attributes
    ----------
    num_classes: int
        Number of classes in the dataset.
    input_channels: int
        Number of channels in the input image.
    learning_rate: float
        Learning rate of the optimizer.
    momentum: float
        Momentum of the optimizer.
    dropout: float
        Dropout probability.

    Methods
    -------
    forward(x: torch.Tensor)
        Forward pass of the network.
    backward(x: torch.Tensor, y: torch.Tensor)
        Backward pass of the network.
    """

    def __init__(
        self,
        num_classes: int = 100,
        input_channels: float = 3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        dropout: float = 0.2,
        learning_decay: float = 0.0005,
    ):
        """
        Neural Network with VGG16 architecture

        Parameters
        ----------
        num_classes: int
            Number of classes in the dataset.
        input_channels: int
            Number of channels in the input image.
        learning_rate: float
            Learning rate of the optimizer.
        momentum: float
            Momentum of the optimizer.
        dropout: float
            Dropout probability.
        learning_decay: float
            Learning decay of the optimizer.
        """
        # Call parent class constructor
        super(RESTNET50, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.learning_decay = learning_decay

        # Define network layers
        # Zero padding
        self.padding = nn.ZeroPad2d((3, 3, 3, 3))

        # Stage 1
        self.stage_1 = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        # Stage 2 (3 blocks of: 1x1 64, 3x3 64, 1x1 256)
        self.stage_2 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=256,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(256),
                    ]
                )
                for _ in range(3)
            ]
        )

        # Stage 3 (4 blocks of: 1x1 128, 3x3 128, 1x1 512)
        self.stage_3 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=512,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(512),
                    ]
                )
                for _ in range(4)
            ]
        )

        # Stage 4 (6 blocks of: 1x1 256, 3x3 256, 1x1 1024)
        self.stage_4 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=256,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=1024,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(1024),
                    ]
                )
                for _ in range(6)
            ]
        )

        # Stage 5 (3 blocks of: 1x1 512, 3x3 512, 1x1 2048)
        self.stage_5 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=1024,
                            out_channels=512,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=512,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=2048,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        nn.BatchNorm2d(2048),
                    ]
                )
                for _ in range(3)
            ]
        )

        # Average pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)

        # Fully connected layers
        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2048 * 1 * 1, 2048),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(2048, num_classes),
                )
            ]
        )

        # Define optimizer
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        # > Loss and cost function
        self.loss = nn.CrossEntropyLoss()
        self.actual_loss = 0
        self.cost = []

        self.actual_corrects = 0
        self.acc = []

        self.epochs_printed = []

    def __str__(self) -> str:
        """
        String representation of the network
        """
        return f"RESNET50(num_classes={self.num_classes},\
            input_channels={self.input_channels},\
            learning_rate={self.learning_rate},\
            momentum={self.momentum},\
            dropout={self.dropout}),\
            \nNum of layers: {len(list(self.parameters()))}\
            \nStage 1: {len(self.stage_1)} layers\
            \nStage 2: {len(self.stage_2)} blocks of 3 layers\
            \nStage 3: {len(self.stage_3)} blocks of 3 layers\
            \nStage 4: {len(self.stage_4)} blocks of 3 layers\
            \nStage 5: {len(self.stage_5)} blocks of 3 layers\
            \nFC: {len(self.fc_layers)} layers"

    def forward(self, x: torch.Tensor, show_img: bool = False) -> torch.Tensor:
        """
        Forward pass of the network

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        show_steps: bool
            Whether to show the output of each layer or not.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        # List to store the output of each layer
        outputs = []
        shortcut = x

        # Forward pass of the network with residual connections
        x = self.padding(x)
        
        # Stage 1
        for layer in self.stage_1:
            x = layer(x)
            outputs.append(x.cpu())
            
        # Stage 2
        for block in self.stage_2:
            for layer in block:
                x = layer(x)
                
                # Check if the dimensions of the input and output are the same
                if x.shape == shortcut.shape:
                    x += shortcut
                else:
                    shortcut = nn.Conv2d(
                        in_channels=shortcut.shape[1],
                        out_channels=x.shape[1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )(shortcut)
                    x += shortcut
                
                x += shortcut
                shortcut = x
            outputs.append(x.cpu())
        
        # Resi
            
        # Stage 3
        for block in self.stage_3:
            for layer in block:
                x = layer(x)
                x += shortcut
                shortcut = x
            outputs.append(x.cpu())
            
        # Stage 4
        for block in self.stage_4:
            for layer in block:
                x = layer(x)
                x += shortcut
                shortcut = x
            outputs.append(x.cpu())
            
        # Stage 5
        for block in self.stage_5:
            for layer in block:
                x = layer(x)
                x += shortcut
                shortcut = x
            outputs.append(x.cpu())
            
        # Plot images in grid
        if show_img:
            fig = plt.figure(figsize=(10, 10))
            for i in range(len(outputs)):
                fig.add_subplot(1, len(outputs), i + 1)
                plt.imshow(outputs[i][0][0].detach().numpy())
                plt.axis("off")
            plt.show()

        x = self.avg_pool(x)

        x = x.view(-1, 2048 * 1 * 1)
        for layer in self.fc_layers:
            x = layer(x)

        return x

    def backward(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> float:
        """
        Backward pass of the network

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        y: torch.Tensor
            Target tensor.
        epoch: int
            Current epoch.

        Returns
        -------
        float
            Loss of the network.
        """
        # Get the prediction
        prediction = self.forward(x)
        
        # Calculate loss
        loss = self.loss(prediction, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Save loss
        self.actual_loss = loss.item()
        
        # Save accuracy
        _, preds = torch.max(prediction, 1)
        
        self.actual_corrects = torch.sum(preds == y)
        
        return loss.item()

        return loss.item()


if __name__ == "__main__":
    # Create network
    net = RESTNET50()
    print(net)
