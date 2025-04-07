from utils import get_train_test
from GAN import Discriminator


BATCH_SIZE = 64
IMAGE_SIZE = 512


print("\nTraining Discriminator:")
full_train_discriminator = Discriminator(image_size=IMAGE_SIZE)
folders = ["inpainting", "insight","text2img"]
for name in folders:
    train_df, test_df = get_train_test(real="data/wiki/", fake=f"data/{name}/")
    
    print(f"\nTraining full Discriminator:")
    full_train_discriminator.fit(train_df, val_split=0.2, epochs=10, batch_size=BATCH_SIZE, lr=0.0002, log_dir=f"runs/full_discriminator_fit_{name}")
    
    print(f"\nTraining Discriminator: '{name}'")
    single_train_discriminator = Discriminator(image_size=IMAGE_SIZE)
    single_train_discriminator.fit(train_df, val_split=0.2, epochs=15, batch_size=BATCH_SIZE, lr=0.0002, log_dir=f"runs/{name}_discriminator")
    single_train_discriminator.save(f"models/discriminator_{name}.pth")
    
full_train_discriminator.save(f"models/discriminator_full.pth")
