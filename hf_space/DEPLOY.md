# Deploy to Hugging Face Spaces

This directory is the deploy-ready bundle. Sayaka pushes manually — the
autonomous session does NOT push (per HANDOFF.md §1).

## One-time setup

1. Create a Hugging Face account if you don't have one, and verify email.
2. Create a Space at https://huggingface.co/new-space
   - Owner: your username
   - Space name: `amp-classifier` (or whatever)
   - License: MIT
   - SDK: Gradio
   - Space hardware: CPU basic (free)
3. Install the Hugging Face CLI:
   ```powershell
   pip install --upgrade huggingface_hub
   huggingface-cli login
   ```
   Paste a write-token from https://huggingface.co/settings/tokens

## Build the deploy bundle

The `hf_space/` directory contains the Space-specific files (README.md,
requirements.txt, this DEPLOY.md). You also need to copy `app.py`, the
`ml/` Python package, and the trained checkpoints into it.

From the project root in PowerShell:

```powershell
cd "C:\Users\danie\Downloads\Claude\projects\AMP Classifier"

# Copy app.py
Copy-Item app.py hf_space\

# Copy the ml/ package (features, scripts, etc.)
Copy-Item -Recurse -Force ml hf_space\ml
# Remove cached / data dirs that aren't needed at inference time
Remove-Item -Recurse -Force hf_space\ml\data -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force hf_space\ml\__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force hf_space\ml\scripts\__pycache__ -ErrorAction SilentlyContinue

# Verify checkpoints are present (the bundle's whole point)
Get-ChildItem hf_space\ml\checkpoints
```

You should see `baseline_xgb.json` (~1-3 MB) and `cnn_best.pt` (~1 MB).
The other baseline checkpoints (LogReg, RF) can be deleted to save space
since the demo only loads XGB + CNN, but it's harmless to leave them.

## Push to the Space

Once your HF Space exists and the bundle is built:

```powershell
cd "C:\Users\danie\Downloads\Claude\projects\AMP Classifier\hf_space"
git init -b main
git remote add origin https://huggingface.co/spaces/<HF_USERNAME>/amp-classifier
git lfs install
git lfs track "*.pt" "*.json"
git add .gitattributes
git add .
git commit -m "Initial AMP Classifier deploy"
git push -u origin main
```

The first push takes a few minutes. Hugging Face will install
`requirements.txt`, then start your app at
`https://huggingface.co/spaces/<HF_USERNAME>/amp-classifier`.

## Verify

The first build of a Space takes ~5-10 min (installing torch, etc.).
Watch the "Logs" tab on the Space page. When it says "Running on local
URL: http://0.0.0.0:7860", the app is live.

Smoke test once it's up:

- **LL-37** (`LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES`) — both models
  should call AMP confidently.
- **Cytochrome-c N-terminus**
  (`GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSY`) — XGBoost is
  fooled by net charge and calls AMP (~0.62); CNN correctly calls
  non-AMP. Honest demonstration of model complementarity.
- **Magainin-2 F12A mutant** (`GIGKFLHSAKKAGKAFVGEIMNS`) — boundary
  case; both models should still call AMP but with less confidence than
  wild-type.

## Update the model card

Once your HF username is set, replace `<HF_USERNAME>` in:

- `hf_space/README.md` (GitHub repo link)
- `docs/model_card.md` (also gets auto-updated by Step 11's docs pass)

## License caveats

DRAMP is **CC-BY-NC for academic use**. Any redistribution of the
classifier in a commercial product would require a separate licensing
discussion with the DRAMP authors. The Space is fine for academic use
and portfolio demonstration. Note this prominently in the Space README
(already done).

## If the deploy fails

Most common cause: the CPU torch wheel didn't resolve. Check the Space
build logs for `Could not find a version that satisfies the requirement
torch`. If so, edit `requirements.txt` to pin torch more loosely
(`torch>=2.4`) or add an explicit `--index-url
https://download.pytorch.org/whl/cpu` line at the top.

Second most common: model checkpoint files weren't tracked by git LFS
and got committed as text. Fix with:

```powershell
git lfs migrate import --include="*.pt,*.json"
git push --force
```

If everything else fails, the bundle is small enough to upload manually
via the HF Space web UI (Files & versions → Add file → Upload files).
