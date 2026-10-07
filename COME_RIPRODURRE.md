# Come riprodurre gli esperimenti

Questa guida spiega come rieseguire la pipeline completa: pre-training continual di FT, FD e
PFR su CIFAR-100 (4 task da 25 classi), l'upper bound joint, le linear evaluation, le metriche CL
e l'indagine su FD. Il percorso principale è il notebook su Google Colab; in fondo c'è
l'equivalente da terminale. Il report è in `report/PFR_report_Papetti.pdf`, i grafici in `figures/`.

## Cosa serve

| Risorsa | Dettagli |
|---|---|
| GPU | una GPU CUDA. Risultati ottenuti su **NVIDIA L4** (Colab); una T4 funziona, più lenta |
| Spazio disco | ~4 GB per i checkpoint (165–315 MB l'uno, 14 in totale) + ~170 MB per CIFAR-100 |
| Account Weights & Biases | necessario per le Sez. 10.1, 11 e per alcuni passi dell'indagine, che leggono le metriche dalle run W&B |
| Tempo | ~12 s/epoca per task su L4 → ~20 min per task a 100 epoche (stime in fondo) |

CIFAR-100 viene scaricato automaticamente da torchvision alla prima esecuzione.

## Percorso 1 — Google Colab (consigliato)

1. Aprire `PFR_riproduzione.ipynb` in Colab e scegliere un runtime GPU
   (`Runtime → Change runtime type`).
2. Eseguire la Sez. 1 in ordine: mount di Drive, clone di questo repository su Drive
   (`/content/drive/MyDrive/CVPR_PFR`, così checkpoint e dati sopravvivono alle disconnessioni),
   dipendenze, login W&B, **patch di compatibilità**, configurazione.
   - Il login W&B è interattivo: la API key non va mai scritta nel notebook. In alternativa
     esportare `WANDB_API_KEY` prima di avviare, oppure usare `WANDB_MODE=offline` e fare
     `wandb sync` alla fine (le celle che leggono da W&B funzionano solo dopo il sync).
   - Le patch vanno rieseguite **a ogni nuovo runtime**: la patch 1 e la 3 modificano file
     dentro l'installazione di Python, che Colab azzera.
3. Eseguire gli esperimenti nell'ordine del notebook:

   | Sezione | Cosa fa | Output su Drive |
   |---|---|---|
   | 5 | pre-training del task 0 (Barlow Twins puro), condiviso da FT, FD e PFR | `experiments/task0/` |
   | 6 | FT — task 1→3 senza distiller | `experiments/ft/` |
   | Joint | upper bound: `main_pretrain.py --num_tasks 1` (tutte le 100 classi insieme) | `experiments/joint/` |
   | 7 | FD — `--distiller decorrelative --distill_lamb 25` | `experiments/fd/` |
   | 8 | PFR — `--distiller pfr` (λ = 1 di default) | `experiments/pfr/` |
   | 9 | linear eval di tutti i checkpoint con `main_linear.py` | run W&B `linear_*` |
   | 10–11 | matrici R, metriche CL, tre varianti di probe e kNN, grafici | figure `.png` |
   | 12 | indagine su FD, incluso il rilancio con `--distill_lamb 5` | `experiments/fd_lamb5/` |

4. **Nomi delle run.** Le celle di analisi ricostruiscono le matrici R cercando le run W&B per
   nome, quindi i nomi vanno lasciati come sono: `linear_t0` (o `linear_task0`), `linear_joint`,
   `linear_{ft,fd,pfr}_T{1,2,3}`, `linear_fd_lamb5_T{1,2,3}`.

Le Sez. 10.2–10.5 e i Passi 2–3 dell'indagine **non usano W&B**: leggono i checkpoint da Drive,
estraggono le feature una volta (cache in `/content/feat_cache`) e allenano probe e kNN in locale.

### Rieseguire solo l'analisi

Se i checkpoint esistono già, il training si salta: basta la Sez. 1 e poi le sezioni dalla 9 in
avanti. La cella 5.6 controlla da sola quali checkpoint di `fd_lamb5` mancano prima di lanciare
qualsiasi training.

## Percorso 2 — Da terminale

```bash
git clone https://github.com/Streaming-Data-Analytics/2025-2026_10b_CL-Code-Reproducibility_PFR.git CVPR_PFR
cd CVPR_PFR
python -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
pip install "pip<24.1"
pip install -r requirements.txt
python scripts/apply_compat_patches.py
wandb login                                             # oppure: export WANDB_MODE=offline
export DATA_DIR=./cifar_data
```

Parametri comuni a tutti i pre-training (identici al notebook):

```bash
COMMON="--dataset cifar100 --encoder resnet18 --data_dir $DATA_DIR --split_strategy class \
  --max_epochs 100 --gpus 0 --num_workers 4 --precision 16 \
  --optimizer sgd --lars --grad_clip_lars --eta_lars 0.02 --exclude_bias_n_norm \
  --scheduler warmup_cosine --lr 0.3 --classifier_lr 0.1 --weight_decay 1e-4 --batch_size 256 \
  --brightness 0.4 --contrast 0.4 --saturation 0.2 --hue 0.1 \
  --gaussian_prob 0.0 0.0 --solarization_prob 0.0 0.2 \
  --method barlow_twins --proj_hidden_dim 2048 --output_dim 2048 --scale_loss 0.1 \
  --save_checkpoint --disable_knn_eval --project pfr --wandb"
```

```bash
# Task 0 condiviso
python main_pretrain.py $COMMON --num_tasks 4 --task_idx 0 --name t0_pretrain \
  --checkpoint_dir ./experiments/task0
T0=$(ls ./experiments/task0/*/*.ckpt | head -n 1)

# FT, FD e PFR: main_continual.py allena in sequenza i task 1, 2 e 3
python main_continual.py $COMMON --num_tasks 4 --task_idx 1 --name ft \
  --checkpoint_dir ./experiments/ft --pretrained_model $T0
python main_continual.py $COMMON --num_tasks 4 --task_idx 1 --name fd \
  --checkpoint_dir ./experiments/fd --pretrained_model $T0 \
  --distiller decorrelative --distill_lamb 25
python main_continual.py $COMMON --num_tasks 4 --task_idx 1 --name pfr \
  --checkpoint_dir ./experiments/pfr --pretrained_model $T0 --distiller pfr

# Joint (upper bound): un unico "task" con tutte le 100 classi
python main_pretrain.py $COMMON --num_tasks 1 --task_idx 0 --name joint_pretrain \
  --checkpoint_dir ./experiments/joint

# Ablation FD con lambda ridotto
python main_continual.py $COMMON --num_tasks 4 --task_idx 1 --name fd_lamb5 \
  --checkpoint_dir ./experiments/fd_lamb5 --pretrained_model $T0 \
  --distiller decorrelative --distill_lamb 5
```

Linear evaluation, da ripetere per ogni checkpoint con il nome di run corrispondente. Anche per il
joint si usa `--num_tasks 4`: stesso seed (5) e stesso `randperm(100).chunk(4)`, quindi le
accuracy per task sono confrontabili con quelle dei metodi continual.

```bash
python main_linear.py --dataset cifar100 --encoder resnet18 --data_dir $DATA_DIR \
  --split_strategy class --max_epochs 100 --gpus 0 --precision 16 \
  --optimizer sgd --scheduler step --lr_decay_steps 60 80 --lr 0.1 --weight_decay 0 \
  --batch_size 256 --num_workers 4 --num_tasks 4 \
  --pretrained_feature_extractor <checkpoint.ckpt> --name linear_ft_T1 --project pfr --wandb
```

Poi aprire il notebook in locale (Jupyter) ed eseguire le sezioni di analisi, dopo aver
aggiornato i percorsi `/content/drive/MyDrive/...` con quelli locali (`DATA_DIR`, `BASE`,
`BASE14`, `BASE_CK`, `EXP_FD5`).

## Tempi indicativi (NVIDIA L4, 100 epoche)

| Passo | Tempo |
|---|---|
| Task 0 | ~20 min |
| FT / FD / PFR (3 task ciascuno) | ~1 h ciascuno, FD un po' più lento |
| Joint (100 classi, 100 epoche) | ~1 h |
| Linear eval `main_linear.py` | ~15 min ciascuna, 11 in totale (+3 per `fd_lamb5`) |
| Probe per experience, Seen/All-Pooled, kNN (feature in cache) | ~30–45 min in tutto |

Le sessioni Colab gratuite possono interrompersi: i checkpoint su Drive permettono di ripartire
dall'ultimo task completato (`main_continual.py --task_idx <t> --pretrained_model <ckpt del task t-1>`).

## Problemi noti

| Sintomo | Causa | Soluzione |
|---|---|---|
| `ModuleNotFoundError: torch._six` | rimosso in PyTorch 2.0 | patch 1 (`scripts/apply_compat_patches.py`) |
| `TypeError: ... compute_on_step` | argomento rimosso da torchmetrics | patch 2 (già applicata in questo fork) |
| `ValueError: The provided lr scheduler ... is invalid` | controllo `isinstance(_LRScheduler)` in PL 1.5.4 | patch 3 |
| `pip` rifiuta pytorch-lightning 1.5.4 | version specifier `torch>=1.7.*` non standard | `pip install "pip<24.1"` prima dei requirements |
| Argomenti vuoti nei comandi `!python ...` (es. `--encoder: expected one argument`) | interpolazione IPython di `$VAR` inaffidabile dentro funzioni | costruire il comando come f-string Python ed eseguirlo con `!{cmd}` (come nel notebook) |
| Run partite ma assenti su W&B dopo una disconnessione | login W&B perso in silenzio | rifare `wandb.login()` prima delle celle con `--wandb` |
