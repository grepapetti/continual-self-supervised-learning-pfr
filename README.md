# 2025-2026: (10b) Revive & Reproduce CSSL Code:
Continually Learning Self-Supervised Representations with PFR

Optional project of the [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=837284&__pj0=0&__pj1=36cd41e96fcd065c47b49d18e46e3110).

Student: **[To be assigned]**

---

# Brief Description

> **Continually Learning Self-Supervised Representations with Projected
> Functional Regularization** — Alex Gomez-Villa et al. (CVPRW, 2022)  
> https://github.com/alviur/CVPR_PFR

The paper proposes **Projected Functional Regularization (PFR)**, a
method that introduces a temporal projection network to map current
latent embeddings back to the previous feature space, addressing
catastrophic forgetting in unsupervised sequential settings without
requiring any replay of past data.

Your objective is threefold: (1) map the theoretical concepts in the
paper to their actual implementation in the codebase; (2) revive the
public repository — written approximately 3–4 years ago — and make it
run in a modern Python environment; (3) reproduce the simplest
experimental case and compare your results against the paper's reported
metrics.

---

# Background

**Self-Supervised Learning (SSL)** learns visual representations without
human annotations by enforcing invariance to data augmentations (SimCLR,
BYOL, BarlowTwins, SimSiam). **Continual Learning (CL)** studies how
models can learn sequentially without catastrophically forgetting previous
knowledge. Their combination — **Continual Self-Supervised Learning
(CSSL)** — is the focus of this project.

PFR addresses the CSSL setting by introducing a temporal projection
network $m$ that maps current embeddings back to the previous feature
space. Unlike direct feature distillation — which penalizes any change
in representations and thus limits plasticity — PFR only constrains
the projection, leaving the backbone free to learn new features. The
regularization loss is:

$$\mathcal{L}_c^t + \lambda_{pfr} \mathbb{E}_{x_a, x_b \sim \mathcal{D}^*_i}
[\mathcal{S}(m(f_{\theta_t}(x_a)), f_{\theta_{t-1}}(x_a))
+ \mathcal{S}(m(f_{\theta_t}(x_b)), f_{\theta_{t-1}}(x_b))]$$

where $\mathcal{S}(\cdot, \cdot)$ is the cosine similarity and
$f_{\theta_{t-1}}$ is the frozen encoder from the previous task.

---

# Project Goals

1. **Map Theory to Code:** Document how the paper's key mechanisms — SSL
objective, temporal projection network, regularization loss — map to
specific files, classes, and functions in the repository.

2. **Revive the Codebase:** Resolve broken dependencies, deprecated APIs,
and missing configurations. Document every issue and solution in the
README of your fork.

3. **Reproduce the Baseline Experiment:** BarlowTwins + CIFAR-100 + 4
tasks + Class-incremental. Simplify if needed (fewer epochs, reduced
tasks) and critically compare results against the paper.

4. **Connect to Course Concepts:** Explicitly map the methods and ideas
in the paper to the concepts covered in the course. For each key
mechanism, discuss similarities and differences with respect to what was
presented in the lectures — motivating why the paper's approach
converges, diverges, or extends the theoretical framework seen in class.

---

# Hyperparameters & Dataset

Both will be **defined together with the project supervisor** at the
start of the project and added to this README before experiments begin.

---

# Evaluation Metrics

Use the standard **linear evaluation** protocol: after training, a frozen
linear classifier is evaluated on top of the learned representations.
Report **Average Accuracy (A)**, **Forgetting (F)**, and **Forward
Transfer (FT)**.

Compare your results across three conditions:
- **Full Training (upper bound):** model trained offline on all data at once.
- **No CL (fine-tuning baseline):** model trained sequentially with no
anti-forgetting strategy.
- **CL technique (PFR):** the method proposed in the paper.

The experimental setup should follow the simplest configuration described
in the paper, without accounting for computational constraints or
execution time — these will be discussed together with the project
supervisor.

As an optional extension, you may also evaluate representation quality
using **KNN classification** or **CKA similarity** (to measure how much
representations shift across tasks).

Present all results in a single table comparing your values against those
reported in the paper.

---

# Deliverables

1. **GitHub Repository** — Fork of the original codebase with an updated
`requirements.txt` and a README documenting all changes and instructions
to reproduce your experiment.

2. **Jupyter Notebook** — End-to-end pipeline: data loading, training,
evaluation, and a results table with plots comparing your metrics against
the paper. Inline comments must make explicit which part of the code
corresponds to which mechanism in the paper, and how it relates to the
concepts covered in the course.

3. **Written Report (~1,500 words)** covering:
   - *What you changed and why:* dependencies fixed, configurations
   adjusted, simplifications made — and the motivation behind each
   choice.
   - *How results compare:* quantitative delta vs. the paper, with a
   reasoned hypothesis for each discrepancy.
   - *Paper-to-code mapping:* for each core contribution (temporal
   projection network, regularization loss, frozen encoder), identify
   exactly where and how it is implemented in the code.
   - *Connections to the course:* for each key concept in the paper,
   discuss explicitly how it relates to what was covered in the lectures
   — highlighting similarities, differences, and extensions.

---

## Note for Students

* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Make any changes to your repository, according to the specific assignment;
* Add a `requirement.txt` file for code reproducibility and instructions on how to replicate the results;
* Commit your changes to your local repository;
* Push your changes to your online repository.
