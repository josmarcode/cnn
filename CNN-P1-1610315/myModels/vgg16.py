import torch
from torch import nn
import matplotlib.pyplot as plt


class VGG16(nn.Module):
    """
    Neural Network based on VGG16 architecture
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
        learning_rate: float = 0.001,
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
        super(VGG16, self).__init__()

        # Initialize attributes
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.learning_decay = learning_decay

        # Define layers based on VGG16 architecture
        # Define a sequential of sequential layers based
        # on the VGG16 layers (2 conv layers + 1 maxpool layer)
        # that are repeated four times and duplicating the
        # number of filters after each maxpool layer (64, 128, 256, 512, 512)
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=x,
                        out_channels=y,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                    nn.BatchNorm2d(y),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=y,
                        out_channels=y,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                    nn.BatchNorm2d(y),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                for x, y in zip(
                    [input_channels, 64, 128, 256, 512], [64, 128, 256, 512, 512]
                )
            ]
        )

        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512 * 1 * 1, 512),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(512, num_classes),
                )
            ]
        )

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
        return f"VGG16(num_classes={self.num_classes},\
            input_channels={self.input_channels},\
            learning_rate={self.learning_rate},\
            momentum={self.momentum},\
            dropout={self.dropout})\
            \n- conv_layers={self.conv_layers},\
            \n- fc_layers={self.fc_layers}"

    def forward(self, x: torch.Tensor, show_steps: bool = False) -> torch.Tensor:
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
        
        # Forward pass of the network
        for layer in self.conv_layers:
            x = layer(x)
            if show_steps:
                outputs.append(x.cpu())
        
        # Plot images in grid
        if show_steps:
            fig = plt.figure(figsize=(10, 10))
            for i in range(len(outputs)):
                fig.add_subplot(1, len(outputs), i + 1)
                plt.imshow(outputs[i][0][0].detach().numpy())
                plt.axis("off")
            plt.show()
        
        x = x.view(-1, 512 * 1 * 1)
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
            Loss value.
        """
        # Get the prediction
        y_pred = self.forward(x, show_steps=epoch not in self.epochs_printed)
        self.epochs_printed.append(epoch)
        
        # Optimize the weights
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Take the maximum prediction as the class
        _, preds = torch.max(y_pred, 1)
        self.actual_corrects += torch.sum(preds == y)
        
        # Accumulate the loss
        self.actual_loss += loss.item() * x.size(0)
        
        return loss.item()


if __name__ == "__main__":
    # Create an instance of the network
    net = VGG16()
    
    # Show the network architecture
    print(net)
