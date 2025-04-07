from GAN import GAN
import pandas as pd
import os


print("\nTraining GAN:")
path = "data/celeba/"
all_files = sorted([
    os.path.join(path, fname)
    for fname in os.listdir(path)
    if fname.endswith(".jpg")
])
df = pd.DataFrame({"path": all_files, "label": 1})

gan = GAN(latent_dim=128)

gan.fit(
    dataframe=df,
    batch_size=512,
    epochs=100,
    lr_gen=0.0003,
    lr_disc=0.0012,
    val_split=0.1,
    label_smoothing=True,
    clip_grad=True,
    max_norm=1.0,
    verbose=True,
    fid_every=5,
    use_instance_noise=True,
    noise_std=0.05,
    initial_gen_steps=2,
    initial_disc_steps=1,
    min_gen_acc=0.65,
    max_disc_acc=0.9,
    gen_patience=3,
    disc_patience=3,
    max_g_steps=5,
    max_d_steps=3,
    scheduler_step=20,
    scheduler_gamma=0.5
)

os.makedirs("models", exist_ok=True)
gan.save()
