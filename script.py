from GAN import GAN, Discriminator
from utils import get_train_test
import os



TRAIN_DISCS = False


TRAIN_GAN = True




BATCH_SIZE = 256
IMAGE_SIZE = 512


if TRAIN_DISCS:
    full_train_discriminator = Discriminator()
    folders = ["inpainting", "insight","text2img"]
    for name in folders:
        train_df, test_df = get_train_test(real="data/wiki/", fake=f"data/{name}/")
        
        print(f"\nTraining full Discriminator:")
        full_train_discriminator.fit(train_df, val_split=0.2, epochs=2, batch_size=BATCH_SIZE, lr=0.0002)
        
        print(f"\nTraining Discriminator: '{name}'")
        single_train_discriminator = Discriminator()
        single_train_discriminator.fit(train_df, val_split=0.2, epochs=6, batch_size=BATCH_SIZE, lr=0.0002)
        single_train_discriminator.save(f"models/discriminator_{name}.pth")
        
    full_train_discriminator.save(f"models/discriminator_full.pth")



def load_latest_gan():
    gan_dirs = [d for d in os.listdir("models") if d.startswith("gan_epoch_")] if os.path.exists("models") else []
    latest_model_path = f"models/{max(gan_dirs, key=lambda x: int(x.split('_')[-1]))}" if gan_dirs else None
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"Loading GAN model from {latest_model_path}...")
        gan = GAN.load("models/gan/", start_epoch=int(latest_model_path.split("_")[-1]))
    else:
        print("Creating new GAN model...")
        gan = GAN(latent_dim=512)
    return gan
    


if TRAIN_GAN:
    print("\nTraining GAN:")
    df, _ = get_train_test(real="data/wiki/")

    gan = load_latest_gan()

    gan.fit(
        dataframe=df,
        batch_size=BATCH_SIZE//2,
        epochs=100,
        lr_gen=0.0002,
        lr_disc=0.00001,
        verbose=True
    )

    os.makedirs("models", exist_ok=True)
    gan.save("models/generator.pth")
