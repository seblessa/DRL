from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from .discriminator import Discriminator
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from .ImageDataset import ImageDataset
from multiprocessing import cpu_count
from torchvision import transforms
from .generator import Generator
import torch.nn as nn
import torch
import sys
import os
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class GAN:
    def __init__(self, latent_dim=256, gen_params=None, disc_params=None):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim, image_size=128, **(gen_params or {}))
        self.discriminator = Discriminator(image_size=128, **(disc_params or {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.fixed_z = torch.randn(16, self.latent_dim).to(self.device)
        self.writer = SummaryWriter(log_dir="runs/gan_training")

    def denormalize(self, images):
        return (images * 0.5) + 0.5

    def fit(self, dataframe, batch_size=128, epochs=25,
        lr_gen=0.0003, lr_disc=0.0012,
        val_split=0.1,
        label_smoothing=True, clip_grad=True, max_norm=1.0,
        verbose=True, fid_every=5,
        use_instance_noise=True, noise_std=0.05,
        initial_gen_steps=2, initial_disc_steps=1,
        min_gen_acc=0.65, max_disc_acc=0.9,
        gen_patience=3, disc_patience=3,
        max_g_steps=5, max_d_steps=3,
        scheduler_step=20, scheduler_gamma=0.5):

        # Split into training and validation sets (using stratification on 'label')
        train_df, val_df = train_test_split(dataframe, test_size=val_split, stratify=dataframe['label'], random_state=42)
        transform = transforms.Compose([
            transforms.Resize((self.discriminator.image_size, self.discriminator.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        train_set = ImageDataset(train_df, transform)
        val_set = ImageDataset(val_df, transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True)

        criterion = nn.BCELoss()
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))

        # Set up learning rate schedulers
        gen_scheduler = StepLR(opt_gen, step_size=scheduler_step, gamma=scheduler_gamma)
        disc_scheduler = StepLR(opt_disc, step_size=scheduler_step, gamma=scheduler_gamma)

        # Initialize dynamic training steps and patience counters
        gen_steps = initial_gen_steps
        disc_steps = initial_disc_steps
        gen_patience_counter = 0
        disc_patience_counter = 0

        outer = tqdm(range(epochs), desc="Epochs", disable=not verbose)
        for epoch in outer:
            self.generator.train()
            self.discriminator.train()

            total_d_loss, total_g_loss = 0.0, 0.0
            total_correct_real, total_correct_fake, total_samples = 0, 0, 0

            inner = tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False, disable=not verbose)
            for batch_idx, (real_imgs, _) in inner:
                real_imgs = real_imgs.to(self.device)
                bs = real_imgs.size(0)
                total_samples += bs

                # Optionally add instance noise to real images
                if use_instance_noise:
                    real_imgs = real_imgs + noise_std * torch.randn_like(real_imgs)

                real_val = 0.9 if label_smoothing else 1.0
                real_targets = torch.full((bs, 1), real_val, device=self.device)
                fake_targets = torch.zeros((bs, 1), device=self.device)

                # --- Discriminator Training ---
                for _ in range(disc_steps):
                    preds_real = self.discriminator(real_imgs)
                    loss_real = criterion(preds_real, real_targets)

                    noise = torch.randn(bs, self.latent_dim, device=self.device)
                    fake_imgs = self.generator(noise)
                    if use_instance_noise:
                        fake_imgs = fake_imgs + noise_std * torch.randn_like(fake_imgs)
                    preds_fake = self.discriminator(fake_imgs.detach())
                    loss_fake = criterion(preds_fake, fake_targets)

                    loss_disc = (loss_real + loss_fake) / 2

                    opt_disc.zero_grad()
                    loss_disc.backward()
                    if clip_grad:
                        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm)
                    opt_disc.step()

                # --- Generator Training ---
                loss_gen_total = 0.0
                gen_scores_total = 0.0
                for _ in range(gen_steps):
                    noise = torch.randn(bs, self.latent_dim, device=self.device)
                    fake_imgs = self.generator(noise)
                    preds = self.discriminator(fake_imgs)
                    loss_gen = criterion(preds, real_targets)

                    opt_gen.zero_grad()
                    loss_gen.backward()
                    if clip_grad:
                        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm)
                    opt_gen.step()

                    loss_gen_total += loss_gen.item()
                    gen_scores_total += preds.mean().item()

                avg_loss_gen = loss_gen_total / gen_steps

                total_d_loss += loss_disc.item()
                total_g_loss += avg_loss_gen
                total_correct_real += (preds_real > 0.5).float().sum().item()
                total_correct_fake += (preds_fake < 0.5).float().sum().item()

                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar("Batch Loss/Discriminator", loss_disc.item(), global_step)
                self.writer.add_scalar("Batch Loss/Generator", avg_loss_gen, global_step)
                inner.set_postfix(d_loss=loss_disc.item(), g_loss=avg_loss_gen)

            # --- End of Epoch Metrics ---
            avg_d_loss = total_d_loss / len(train_loader) / disc_steps
            avg_g_loss = total_g_loss / len(train_loader) / gen_steps
            real_acc = total_correct_real / total_samples
            fake_acc = total_correct_fake / total_samples
            gen_acc = fake_acc  # Generator success rate: higher means it's fooling D
            disc_acc = (real_acc + (1 - fake_acc)) / 2

            self.writer.add_scalar("Epoch Loss/Discriminator", avg_d_loss, epoch)
            self.writer.add_scalar("Epoch Loss/Generator", avg_g_loss, epoch)
            self.writer.add_scalar("Disc Accuracy/Real", real_acc, epoch)
            self.writer.add_scalar("Disc Accuracy/Fake", fake_acc, epoch)
            self.writer.add_scalar("Training Steps/Generator", gen_steps, epoch)
            self.writer.add_scalar("Training Steps/Discriminator", disc_steps, epoch)
            self.writer.add_scalar("Training Steps/Gen_vs_Disc_Ratio", gen_steps/(disc_steps+1e-6), epoch)

            # --- Dynamic Adjustment ---
            # If the generator is too powerful (gen_acc > 0.9), reduce gen_steps.
            if gen_acc > 0.9 and gen_steps > initial_gen_steps:
                gen_steps = max(initial_gen_steps, gen_steps - 1)
                gen_patience_counter = 0
            elif gen_acc < min_gen_acc and gen_steps < max_g_steps:
                gen_steps += 1
                gen_patience_counter = 0
            else:
                gen_patience_counter += 1
                if gen_patience_counter >= gen_patience and gen_steps > initial_gen_steps:
                    gen_steps = max(initial_gen_steps, gen_steps - 1)
                    gen_patience_counter = 0

            # For discriminator, if its accuracy on real images is too low (<0.7), increase disc_steps.
            if real_acc < 0.7 and disc_steps < max_d_steps:
                disc_steps += 1
                disc_patience_counter = 0
            elif real_acc > max_disc_acc and disc_steps > initial_disc_steps:
                disc_steps = max(initial_disc_steps, disc_steps - 1)
                disc_patience_counter = 0
            else:
                disc_patience_counter += 1
                if disc_patience_counter >= disc_patience and disc_steps > initial_disc_steps:
                    disc_steps = max(initial_disc_steps, disc_steps - 1)
                    disc_patience_counter = 0

            self.writer.add_scalar("Dynamic/Generator Steps", gen_steps, epoch)
            self.writer.add_scalar("Dynamic/Discriminator Steps", disc_steps, epoch)

            # --- Validation Metrics ---
            self.discriminator.eval()
            with torch.no_grad():
                val_loss_d, val_loss_g = 0.0, 0.0
                for val_imgs, _ in val_loader:
                    val_imgs = val_imgs.to(self.device)
                    bs = val_imgs.size(0)
                    real_targets_val = torch.ones((bs, 1), device=self.device)
                    fake_targets_val = torch.zeros((bs, 1), device=self.device)
                    preds_real_val = self.discriminator(val_imgs)
                    loss_val_real = criterion(preds_real_val, real_targets_val)
                    noise = torch.randn(bs, self.latent_dim, device=self.device)
                    fake_imgs_val = self.generator(noise)
                    preds_fake_val = self.discriminator(fake_imgs_val)
                    loss_val_fake = criterion(preds_fake_val, fake_targets_val)
                    val_loss_d += ((loss_val_real + loss_val_fake) / 2).item()
                    val_loss_g += criterion(self.discriminator(fake_imgs_val), real_targets_val).item()
                val_loss_d /= len(val_loader)
                val_loss_g /= len(val_loader)
            self.writer.add_scalar("Val Loss/Discriminator", val_loss_d, epoch)
            self.writer.add_scalar("Val Loss/Generator", val_loss_g, epoch)

            # --- Log Learning Rates via Schedulers ---
            gen_scheduler.step()
            disc_scheduler.step()
            self.writer.add_scalar("LR/Generator", opt_gen.param_groups[0]['lr'], epoch)
            self.writer.add_scalar("LR/Discriminator", opt_disc.param_groups[0]['lr'], epoch)

            # --- Log Generated Images ---
            self.generator.eval()
            with torch.no_grad():
                preview_imgs = self.generator(self.fixed_z).cpu()
            grid = make_grid(self.denormalize(preview_imgs), nrow=4)
            self.writer.add_image("Generated Images", grid, global_step=epoch)

            # --- Save Model & Preview ---
            save_path = f"models/gan/gan_epoch_{epoch}/"
            self.save(save_path)
            transforms.ToPILImage()(grid).save(f"{save_path}generated_images.png")

            # --- FID Evaluation ---
            if (epoch + 1) % fid_every == 0:
                fid = FrechetInceptionDistance().to(self.device)
                fid.reset()
                for real_batch, _ in val_loader:
                    real_uint8 = (self.denormalize(real_batch) * 255).clamp(0, 255).to(torch.uint8)
                    fid.update(real_uint8.to(self.device), real=True)
                    break
                with torch.no_grad():
                    fake_batch = self.generator(torch.randn(batch_size, self.latent_dim).to(self.device))
                fake_uint8 = (self.denormalize(fake_batch) * 255).clamp(0, 255).to(torch.uint8)
                fid.update(fake_uint8.to(self.device), real=False)
                fid_score = fid.compute().item()
                self.writer.add_scalar("FID Score", fid_score, global_step=epoch)

        self.writer.close()


    def save(self, gan_path="models/gan/"):
        os.makedirs(gan_path, exist_ok=True)
        self.generator.save(os.path.join(gan_path, "generator.pth"))
        self.discriminator.save(os.path.join(gan_path, "discriminator.pth"))

    @classmethod
    def load(cls, gan_path="models/gan/"):
        generator_path = os.path.join(gan_path, "generator.pth")
        discriminator_path = os.path.join(gan_path, "discriminator.pth")

        gen_checkpoint = torch.load(generator_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        disc_checkpoint = torch.load(discriminator_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        gen_params = {k: gen_checkpoint[k] for k in gen_checkpoint if k != 'model_state_dict'}
        disc_params = {k: disc_checkpoint[k] for k in disc_checkpoint if k != 'model_state_dict'}

        gan = cls(latent_dim=gen_params.get("latent_dim", 256),
                gen_params=gen_params,
                disc_params=disc_params,)

        # Load weights
        gan.generator.load_state_dict(gen_checkpoint['model_state_dict'])
        gan.discriminator.load_state_dict(disc_checkpoint['model_state_dict'])

        return gan
