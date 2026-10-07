"""Patch di compatibilità per far girare il codice del 2022 su uno stack moderno.

Sono le stesse tre patch della Sez. 1.3 del notebook, raccolte in uno script per chi
lancia gli esperimenti da terminale. Lo script è idempotente: si può rieseguire senza
effetti collaterali. Va lanciato DOPO `pip install -r requirements.txt`, dalla root del repo:

    python scripts/apply_compat_patches.py

Nessuna delle tre patch tocca i calcoli (loss, distiller, ottimizzatore): intervengono
solo su import e controlli di tipo. La verifica numerica è nel notebook, Passo 4.
"""
import inspect
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patch_torch_six():
    """Patch 1: torch._six è stato rimosso in PyTorch 2.0, ma pytorch-lightning 1.5.4 lo importa."""
    six_path = os.path.join(os.path.dirname(torch.__file__), "_six.py")
    if os.path.exists(six_path):
        print("[1/3] torch._six già presente")
        return
    with open(six_path, "w") as f:
        f.write("string_classes = (str,)\n")
    print(f"[1/3] creato {six_path}")


def patch_knn():
    """Patch 2: compute_on_step è stato rimosso da torchmetrics.

    In questo fork il file è già corretto; la patch serve se si parte dal repo originale.
    """
    knn_path = os.path.join(REPO_ROOT, "cassle", "utils", "knn.py")
    with open(knn_path) as f:
        src = f.read()
    old = "dist_sync_on_step=dist_sync_on_step, compute_on_step=False"
    if old not in src:
        print("[2/3] knn.py già compatibile")
        return
    with open(knn_path, "w") as f:
        f.write(src.replace(old, "dist_sync_on_step=dist_sync_on_step"))
    print("[2/3] knn.py patchato")


def patch_lightning_scheduler_check():
    """Patch 3: PL 1.5.4 accetta solo scheduler che ereditano da _LRScheduler.

    Nelle versioni recenti di PyTorch gli scheduler usati dal repo non superano più quel
    controllo: lo sostituisco con un controllo duck-typing (ha step() e optimizer).
    """
    try:
        import pytorch_lightning.trainer.optimizers as opt_mod
    except ImportError as e:  # tipicamente: torch._six mancante -> eseguire prima la patch 1
        sys.exit(f"[3/3] impossibile importare pytorch_lightning: {e}")
    opt_path = inspect.getfile(opt_mod)
    with open(opt_path) as f:
        src = f.read()
    old = (
        "                elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):\n"
        '                    lr_schedulers.append({**default_config, "scheduler": scheduler})'
    )
    new = (
        "                elif hasattr(scheduler, 'step') and hasattr(scheduler, 'optimizer'):\n"
        '                    lr_schedulers.append({**default_config, "scheduler": scheduler})'
    )
    if new in src:
        print("[3/3] pytorch-lightning già patchato")
    elif old in src:
        with open(opt_path, "w") as f:
            f.write(src.replace(old, new))
        print(f"[3/3] patchato {opt_path}")
    else:
        sys.exit("[3/3] pattern non trovato: la versione di pytorch-lightning non è la 1.5.4?")


if __name__ == "__main__":
    patch_torch_six()
    patch_knn()
    patch_lightning_scheduler_check()
    print("Patch applicate.")
