from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .ImageDataset import ImageDataset
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from PIL import Image
import torch
import sys
import os
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self,
                 image_size,
                 input_channels=3,
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

        # Step 1: Build convolutional layers
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

        # Step 2: Compute final feature size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            dummy_out = nn.Sequential(*conv_layers)(dummy)
            final_feat_size = dummy_out.view(1, -1).shape[1]

        # Step 3: Classifier
        classifier = [nn.Flatten(), nn.Linear(final_feat_size, 1)]
        if final_activation == 'sigmoid':
            classifier.append(nn.Sigmoid())

        self.model = nn.Sequential(*conv_layers, *classifier)
        self.to(self.device)

    def fit(self, train_df, val_split=0.1, batch_size=64, epochs=10, lr=0.0002, verbose=True, log_dir="runs/discriminator_training"):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        train_data, val_data = train_test_split(train_df, test_size=val_split, stratify=train_df['label'], random_state=42)
        train_dataset = ImageDataset(train_data, transform)
        val_dataset = ImageDataset(val_data, transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=log_dir)

        outer = tqdm(range(epochs), desc="Epochs", disable=not verbose)
        for epoch in outer:
            self.train()
            running_loss = 0.0
            total_batches = len(train_loader)
            batch_idx = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self(imgs)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                global_step = epoch * total_batches + batch_idx
                writer.add_scalar("Train/Batch Loss", loss.item(), global_step)
                batch_idx += 1

            avg_train_loss = running_loss / total_batches
            writer.add_scalar("Train/Epoch Loss", avg_train_loss, epoch)

            # Validation loop
            self.eval()
            running_val_loss = 0.0
            all_preds = []
            all_labels = []
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    preds = self(imgs)
                    loss = criterion(preds, labels)
                    running_val_loss += loss.item()

                    preds_binary = (preds > 0.5).float()
                    correct += (preds_binary == labels).sum().item()
                    total += labels.size(0)

                    # Accumulate for confusion matrix
                    all_preds.extend(preds_binary.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())

            avg_val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            writer.add_scalar("Val/Epoch Loss", avg_val_loss, epoch)
            writer.add_scalar("Val/Accuracy", val_accuracy, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            outer.set_postfix(loss=avg_train_loss, val_acc=val_accuracy)

            # Compute and log confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            writer.add_figure("Confusion Matrix", fig, global_step=epoch)
            plt.close(fig)

        writer.close()

    
    
    def forward(self, x):
        x = self.model(x)
        return x
            
    def predict(self, test_df):
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
