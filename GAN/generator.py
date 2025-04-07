import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        output_channels=3,
        image_size=128,
        hidden_layers=[512, 256, 128, 64, 32],
        starting_res=4,
        kernel_size=3,
        stride=2,
        padding=1,
        use_batchnorm=True,
        use_spectralnorm=True,
        use_dropout=True,
        dropout_rate=0.15,
        final_activation='tanh'
    ):
        super().__init__()

        assert image_size % starting_res == 0, "image_size must be divisible by starting_res"
        num_upsamples = len(hidden_layers[1:]) + 1
        final_resolution = starting_res * (stride ** num_upsamples)
        if final_resolution != image_size:
            raise ValueError(
                f"[Generator] Configuration mismatch:\n"
                f"  starting_res={starting_res}, stride={stride}, upsample_steps={num_upsamples} → "
                f"final resolution = {final_resolution} ≠ image_size = {image_size}\n\n"
                f"  ➤ Either increase hidden_layers or adjust starting_res/stride"
            )

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.hidden_layers = hidden_layers
        self.starting_res = starting_res
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_spectralnorm = use_spectralnorm
        self.dropout_rate = dropout_rate
        self.final_activation = final_activation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear = nn.Linear(latent_dim, hidden_layers[0] * starting_res * starting_res)

        layers = []
        in_channels = hidden_layers[0]

        for out_channels in hidden_layers[1:]:
            layers.append(nn.Upsample(scale_factor=stride, mode='nearest'))

            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
            if use_spectralnorm:
                conv = nn.utils.spectral_norm(conv)

            layers.append(conv)

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU(inplace=True))

            if use_dropout:
                layers.append(nn.Dropout2d(p=dropout_rate))

            in_channels = out_channels

        # Final layer to output image
        layers.append(nn.Upsample(scale_factor=stride, mode='nearest'))
        final_conv = nn.Conv2d(in_channels, output_channels, kernel_size, stride=1, padding=padding)
        layers.append(final_conv)

        if final_activation == 'tanh':
            layers.append(nn.Tanh())
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), self.hidden_layers[0], self.starting_res, self.starting_res)
        return self.model(x)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith('.pth'):
            path += '.pth'

        torch.save({
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'output_channels': self.output_channels,
            'image_size': self.image_size,
            'hidden_layers': self.hidden_layers,
            'starting_res': self.starting_res,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'use_batchnorm': self.use_batchnorm,
            'use_spectralnorm': self.use_spectralnorm,
            'use_dropout': self.use_dropout,
            'dropout_rate': self.dropout_rate,
            'final_activation': self.final_activation
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model = cls(**{k: checkpoint[k] for k in checkpoint if k != 'model_state_dict'})
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.device)
        return model



    def generate(self, num_images, save=None):
        noise = torch.randn(num_images, self.latent_dim).to(self.device)
        with torch.no_grad():
            generated_images = self(noise)

        fig = plt.figure(figsize=(8, 8))
        for i, img_tensor in enumerate(generated_images):
            img = TF.to_pil_image((img_tensor.cpu() + 1) / 2)
            ax = fig.add_subplot(int(num_images**0.5), int(num_images**0.5), i + 1)
            ax.imshow(img)
            ax.set_title(f"Image {i+1}")
            ax.axis("off")
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save)
        plt.show()
