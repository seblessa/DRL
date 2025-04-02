from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch
import os
import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ----------------------------------------------------------------------
# ImageDataset for loading images from the DataFrame
# ----------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        label = torch.tensor([row['label']], dtype=torch.float32)
        return img, label

# ----------------------------------------------------------------------
# Generator Network
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        output_channels=3,
        image_size=512,
        hidden_layers=[512, 256, 128, 64, 32, 16],
        starting_res=8,
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
        x = x.view(z.size(0), self.hidden_layers[0], self.starting_res, self.starting_res).to(self.device)
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

    def generate(self, num_images):
        noise = torch.randn(num_images, self.latent_dim).to(self.device)
        with torch.no_grad():
            return self(noise)


# ----------------------------------------------------------------------
# Discriminator Network
# ----------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self,
                 input_channels=3,
                 image_size=512,
                 hidden_layers=[64, 128, 256, 512],
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 use_batchnorm=True,
                 final_activation='sigmoid',
                 use_dropout=True,
                 dropout_rate=0.3,
                 use_spectral_norm=True):
        super().__init__()

        self.input_channels = input_channels
        self.image_size = image_size
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_spectral_norm = use_spectral_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conv_layers = []
        in_channels = input_channels
        for i, out_channels in enumerate(hidden_layers):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            conv_layers.append(conv)
            if i != 0 and use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if use_dropout:
                conv_layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            dummy_out = nn.Sequential(*conv_layers)(dummy)
            final_feat_size = dummy_out.view(1, -1).shape[1]

        classifier = [nn.Flatten(), nn.Linear(final_feat_size, 1)]
        if final_activation == 'sigmoid':
            classifier.append(nn.Sigmoid())

        self.model = nn.Sequential(*conv_layers, *classifier)
        self.to(self.device)

    def fit(self, train_df, val_split=0.1, batch_size=64, epochs=10, lr=0.0002, verbose=True):
        from sklearn.model_selection import train_test_split
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        train_data, val_data = train_test_split(train_df, test_size=val_split, stratify=train_df['label'], random_state=42)
        train_dataset = ImageDataset(train_data, transform)
        val_dataset = ImageDataset(val_data, transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        outer = tqdm(range(epochs), desc="Epochs", disable=not verbose)
        for epoch in outer:
            self.train()
            running_loss = 0.0
            inner = tqdm(train_loader, desc="Batches", leave=False, disable=not verbose)
            for imgs, labels in inner:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self(imgs)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                inner.set_postfix(loss=loss.item())
            self.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    preds = self(imgs)
                    preds_binary = (preds > 0.5).float()
                    correct += (preds_binary == labels).sum().item()
                    total += labels.size(0)
            val_acc = 100 * correct / total
            outer.set_postfix(loss=running_loss / len(train_loader), val_acc=val_acc)
            
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
            
    def predict(self, test_df):
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        test_dataset = ImageDataset(test_df, transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        self.eval()
        predictions = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(self.device)
                preds = self(imgs)
                predictions.append(preds.item())
        results_df = test_df.copy()
        results_df['predictions'] = [1 if pred > 0.5 else 0 for pred in predictions]
        results_df['predictions_proba'] = predictions
        return results_df

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not path.endswith('.pth'):
            path += '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'image_size': self.image_size,
            'hidden_layers': self.hidden_layers,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'use_batchnorm': self.use_batchnorm,
            'final_activation': self.final_activation,
            'use_dropout': self.use_dropout,
            'dropout_rate': self.dropout_rate,
            'use_spectral_norm': self.use_spectral_norm
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        defaults = {
            'use_dropout': False,
            'dropout_rate': 0.0,
            'use_spectral_norm': False
        }
        params = {**defaults, **{k: checkpoint.get(k, defaults.get(k)) for k in checkpoint if k != 'model_state_dict'}}
        model = cls(**params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.device)
        return model

    def predict_single(self, image_path):
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        pred = self(input_tensor).item()
        pred_label = 1 if pred > 0.5 else 0
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {pred_label} ({pred:.2f})")
        plt.show()
        return pred_label, pred


# ----------------------------------------------------------------------
# GAN Class – Combines Generator and Discriminator
# ----------------------------------------------------------------------
class GAN:
    def __init__(self, latent_dim=512, gen_params=None, disc_params=None, start_epoch=0):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim, **(gen_params or {}))
        self.discriminator = Discriminator(**(disc_params or {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.start_epoch = start_epoch
        self.fixed_z = torch.randn(16, self.latent_dim).to(self.device)
        self.writer = SummaryWriter(log_dir="runs/gan_training")

    def denormalize(self, images):
        return (images * 0.5) + 0.5

    def fit(self, dataframe, batch_size=64, epochs=25,
            lr_gen=0.0002, lr_disc=0.00005,
            label_smoothing=True, clip_grad=True, max_norm=1.0,
            gen_steps_per_disc=2, verbose=True):
        
        if self.start_epoch >= epochs:
            print("Training already completed.")
            return

        transform = transforms.Compose([
            transforms.Resize((self.discriminator.image_size, self.discriminator.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        dataset = ImageDataset(dataframe, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        criterion = nn.BCELoss()
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))

        outer = tqdm(range(self.start_epoch, epochs), desc="Epochs", disable=not verbose)

        for epoch in outer:
            self.generator.train()
            self.discriminator.train()

            inner = tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches", leave=False, disable=not verbose)

            for batch_idx, (real_imgs, _) in inner:
                real_imgs = real_imgs.to(self.device)
                bs = real_imgs.size(0)

                real_val = 0.9 if label_smoothing else 1.0
                real_targets = torch.full((bs, 1), real_val, device=self.device)
                fake_targets = torch.zeros((bs, 1), device=self.device)

                # ----------------- Discriminator -----------------
                preds_real = self.discriminator(real_imgs)
                loss_real = criterion(preds_real, real_targets)

                noise = torch.randn(bs, self.latent_dim, device=self.device)
                fake_imgs = self.generator(noise)
                preds_fake = self.discriminator(fake_imgs.detach())
                loss_fake = criterion(preds_fake, fake_targets)

                loss_disc = (loss_real + loss_fake) / 2

                opt_disc.zero_grad()
                loss_disc.backward()
                if clip_grad:
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm)
                opt_disc.step()

                # ----------------- Generator -----------------
                for _ in range(gen_steps_per_disc):
                    noise = torch.randn(bs, self.latent_dim, device=self.device)
                    fake_imgs = self.generator(noise)
                    preds = self.discriminator(fake_imgs)
                    loss_gen = criterion(preds, real_targets)

                    opt_gen.zero_grad()
                    loss_gen.backward()
                    if clip_grad:
                        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm)
                    opt_gen.step()

                if verbose:
                    inner.set_postfix(d_loss=loss_disc.item(), g_loss=loss_gen.item())

                global_step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar("Batch Loss/Discriminator", loss_disc.item(), global_step)
                self.writer.add_scalar("Batch Loss/Generator", loss_gen.item(), global_step)

            # ----------------- Epoch Logging -----------------
            self.writer.add_scalar("Epoch Loss/Discriminator", loss_disc.item(), epoch)
            self.writer.add_scalar("Epoch Loss/Generator", loss_gen.item(), epoch)

            d_real_acc = (preds_real > 0.5).float().mean().item()
            d_fake_acc = (preds_fake < 0.5).float().mean().item()
            self.writer.add_scalar("Disc Accuracy/Real", d_real_acc, epoch)
            self.writer.add_scalar("Disc Accuracy/Fake", d_fake_acc, epoch)

            self.generator.eval()
            with torch.no_grad():
                preview_imgs = self.generator(self.fixed_z).cpu()
            grid = make_grid(self.denormalize(preview_imgs), nrow=4)
            self.writer.add_image("Generated Images", grid, global_step=epoch)

            # Save model & grid image
            self.save(f"models/gan_epoch_{epoch}/")
            grid_img = transforms.ToPILImage()(grid)
            grid_img.save(f"models/gan_epoch_{epoch}/generated_images.png")

        self.writer.close()

    def save(self, gan_path="models/gan/"):
        os.makedirs(gan_path, exist_ok=True)
        self.generator.save(os.path.join(gan_path, "generator.pth"))
        self.discriminator.save(os.path.join(gan_path, "discriminator.pth"))

    @classmethod
    def load(cls, gan_path="models/gan/", start_epoch=0):
        generator_path = os.path.join(gan_path, "generator.pth")
        discriminator_path = os.path.join(gan_path, "discriminator.pth")

        gen_checkpoint = torch.load(generator_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        disc_checkpoint = torch.load(discriminator_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        gen_params = {k: gen_checkpoint[k] for k in gen_checkpoint if k != 'model_state_dict'}
        disc_params = {k: disc_checkpoint[k] for k in disc_checkpoint if k != 'model_state_dict'}

        gan = cls(latent_dim=gen_params.get("latent_dim", 256),
                gen_params=gen_params,
                disc_params=disc_params,
                start_epoch=start_epoch)

        # Load weights
        gan.generator.load_state_dict(gen_checkpoint['model_state_dict'])
        gan.discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
        return gan

