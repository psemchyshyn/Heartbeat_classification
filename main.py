from lightning import LitECG
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml



if __name__ == "__main__":
    seed_everything(42)
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    lit = LitECG(config)


    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="checkpoints_rnn",
        filename="model-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    estopping_callback = EarlyStopping(monitor="valid_loss", patience=3)

    trainer = Trainer(
                    logger=WandbLogger(project="mit-bih-classification"), 
                    enable_progress_bar=True,
                    max_epochs=30,
                    check_val_every_n_epoch=1,
                    num_sanity_val_steps=1,
                    callbacks=[checkpoint_callback, estopping_callback])

    trainer.fit(lit)
    trainer.test(lit, ckpt_path="best")
