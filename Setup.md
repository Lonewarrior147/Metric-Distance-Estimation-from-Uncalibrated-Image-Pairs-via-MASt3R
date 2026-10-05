# Setup - Kaggle and Google Colab

This project runs on **Kaggle** or **Google Colab**, and we switch between them when one
runs out of free GPU quota. Both use the same notebook, `main.ipynb`. Two setup cells sit at
the top of it: one for Colab and one for Kaggle. Each cell skips itself on the other platform,
so you can run both without harm.

---

## 0. One-time prerequisites (both platforms)

1. **The project must be on GitHub** and you need the clone URL, for example
   `https://github.com/Lonewarrior147/Metric-Distance-Estimation-from-Uncalibrated-Image-Pairs-via-MASt3R.git`.
   - Put this URL in the `REPO_URL` line of the **Colab** setup cell and the **Kaggle** setup
     cell. It is already set to the project repository.
   - If the repository is **private**, create a GitHub token (fine-grained, read and write
     access to this repo) and see section 4.
2. **A GPU runtime.** MASt3R is too slow on CPU. Use a Tesla T4 on both platforms.

---

## 1. Google Colab

### 1.1 Open the notebook
1. Go to https://colab.research.google.com.
2. **File > Open notebook > GitHub**, paste the repo URL, and open `main.ipynb`.
   (Or upload the file manually.)

### 1.2 Select a GPU
**Runtime > Change runtime type > Hardware accelerator: T4 GPU > Save.**

### 1.3 Run the setup
1. Run the **Colab setup cell** (the first code cell, under the title).
2. It will:
   - clone the project into `/content/mast3r_metric_distance`,
   - check that the GPU is visible,
   - install the three small packages (`roma`, `einops`, `trimesh`).
3. Then run the notebook cells in order, starting with Phase 1.

### 1.4 Important Colab facts
- `/content` is **wiped when the runtime disconnects or resets**. The project clone, the
  checkpoint and any results disappear with it.
- The checkpoint is about 2.75 GB and downloads on every new runtime. Expect a few minutes.
- Free Colab has a daily GPU limit and can disconnect after idle time. Save work regularly
  (section 3).

---

## 2. Kaggle

### 2.1 Open the notebook
1. Go to https://www.kaggle.com/code and **New Notebook**.
2. **File > Import notebook > GitHub** (or upload `main.ipynb`).

### 2.2 Settings (must be done first)
Right sidebar, **Session options**:
- **Accelerator: GPU T4 x1** (or GPU P100 if T4 is unavailable - the code works on either).
- **Internet: On.** Without this, the clone and checkpoint download fail.
- **Persistence: Files only** if the option is offered. Otherwise, rely on git (section 3).

### 2.3 Run the setup
1. Run the **Kaggle setup cell** (the second code cell, under the title).
2. It will:
   - clone the project into `/kaggle/working/mast3r_metric_distance`,
   - check that the GPU is visible,
   - install the three small packages.
3. Then run the notebook cells in order, starting with Phase 1.

### 2.4 Important Kaggle facts
- `/kaggle/working` is the writable directory. Anything outside it is lost.
- Kaggle sessions have a weekly GPU quota, and sessions time out. Save work regularly
  (section 3).
- Kaggle's Python is 3.12. Colab's Python may differ. Phase 1 prints the versions, so check
  them after the first run on each platform.

---

## 3. Saving work before a session ends (both platforms)

Runtime files are temporary, so push results to git before stopping:

```bash
cd <project folder>            # /content/mast3r_metric_distance or /kaggle/working/mast3r_metric_distance
git add main.ipynb results/ CLAUDE.md
git commit -m "Phase N: <what you did>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
git pull --rebase
git push
```

Do **not** add `checkpoints/`, `mast3r/`, `data/` or `.venv/`. They are already in `.gitignore`.

Update your entry in the work log in `CLAUDE.md` before the final push, so the next person
knows where things stand.

---

## 4. Git access from inside Colab or Kaggle (private repo, or pushing from the runtime)

Never paste a token into a notebook cell. Store it as a secret instead.

- **Colab:** the key icon in the left sidebar > add a secret named `GITHUB_TOKEN`, and turn on
  notebook access. Read it in code with `google.colab.userdata.get("GITHUB_TOKEN")`.
- **Kaggle:** **Add-ons > Secrets** > add a secret named `GITHUB_TOKEN`. Read it with
  `kaggle_secrets.UserSecretsClient().get_secret("GITHUB_TOKEN")`.

Then set the clone URL to use it, for example
`https://<TOKEN>@github.com/Lonewarrior147/Metric-Distance-Estimation-from-Uncalibrated-Image-Pairs-via-MASt3R.git`, inside the setup cell only
for that session. Do not print the URL. Do not save it in any file.

Set your git identity once per runtime:

```bash
git config --global user.name "<your name>"
git config --global user.email "<your email>"
```

---

## 5. Switching from Colab to Kaggle (or back)

Use this when a runtime runs out of quota mid-work.

1. **On the old platform:** save and push (section 3). Make sure the last commit is pushed.
2. **On the new platform:** open the notebook and run that platform's setup cell. It clones the
   latest `main`.
3. Check `CLAUDE.md` for the last log entry and the "Current claims" section. Continue from the
   "Next" line.
4. Re-run the notebook cells from the top to rebuild state. Nothing from the old runtime is
   carried over except what was pushed.

---

## 6. Checklist for a fresh runtime

- [ ] GPU is on (T4) and the setup cell printed `CUDA available: True`
- [ ] Project cloned, and `git log -1` matches the latest pushed commit
- [ ] Setup cell passed (Internet on for Kaggle)
- [ ] Phase 1 cells run in order; checkpoint reports byte-exact
- [ ] `CLAUDE.md` read; you have claimed the phase you are working on
