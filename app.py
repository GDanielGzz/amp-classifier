"""Gradio demo for the AMP Classifier — Predict + Design tabs.

PREDICT tab: three models score a user-supplied peptide in parallel
(XGBoost engineered baseline, Platt-calibrated CNN, ESM2-650M + MLP
head). Includes per-XGBoost-feature attribution.

DESIGN tab (Phase 2D): greedy beam search over single-residue
mutations of a parent AMP, scored by the ESM head. Picks the top-K
mutants the head ranks higher than wild-type. Demonstrates the
"discriminator-in-the-loop" pattern: a strong classifier filters a
simple generator's proposals.

Run: python app.py    (or `dev.bat app`)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml import features as F  # noqa: E402
from ml.scripts.train_cnn import CNN, MAX_LEN, onehot_encode  # noqa: E402
from ml.scripts.design_mutants import (  # noqa: E402
    PARENT_LIBRARY, design_mutants,
)

CHECKPOINTS = PROJECT_ROOT / "ml" / "checkpoints"
XGB_PATH = CHECKPOINTS / "baseline_xgb.json"
CNN_PATH = CHECKPOINTS / "cnn_best.pt"
CALIBRATOR_PATH = CHECKPOINTS / "cnn_calibrator.joblib"
ESM_HEAD_PATH = CHECKPOINTS / "esm_head_best.pt"

CNN_RAW_THRESHOLD = 0.050
MAX_INPUT_LEN = 100

ESM_MODEL_NAME = os.environ.get("ESM_MODEL_NAME", "facebook/esm2_t33_650M_UR50D")


def load_models():
    xgb = cnn = calibrator = esm_tokenizer = esm_backbone = esm_head = esm_device = None

    if XGB_PATH.exists():
        xgb = XGBClassifier()
        xgb.load_model(str(XGB_PATH))
        print("[app] loaded XGBoost")
    if CNN_PATH.exists():
        ckpt = torch.load(CNN_PATH, map_location="cpu", weights_only=False)
        cnn = CNN()
        cnn.load_state_dict(ckpt["state_dict"])
        cnn.eval()
        print("[app] loaded CNN")
    if CALIBRATOR_PATH.exists():
        calibrator = joblib.load(CALIBRATOR_PATH)
        print("[app] loaded Platt calibrator for CNN")
    if ESM_HEAD_PATH.exists():
        try:
            from transformers import AutoTokenizer, AutoModel
            from ml.scripts.train_esm_head import EsmHead
            print(f"[app] loading ESM2 backbone ({ESM_MODEL_NAME}) ...")
            esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            esm_backbone = AutoModel.from_pretrained(ESM_MODEL_NAME).eval()
            head_ckpt = torch.load(ESM_HEAD_PATH, map_location="cpu", weights_only=False)
            in_dim = head_ckpt.get("in_dim", esm_backbone.config.hidden_size)
            esm_head = EsmHead(in_dim=in_dim)
            esm_head.load_state_dict(head_ckpt["state_dict"])
            esm_head.eval()
            esm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            esm_backbone = esm_backbone.to(esm_device)
            esm_head = esm_head.to(esm_device)
            print(f"[app] ESM pipeline ready on {esm_device}")
        except Exception as exc:
            print(f"[app] ESM pipeline load failed: {exc!r}")
            esm_tokenizer = esm_backbone = esm_head = esm_device = None
    return xgb, cnn, calibrator, esm_tokenizer, esm_backbone, esm_head, esm_device


(XGB_MODEL, CNN_MODEL, CALIBRATOR,
 ESM_TOKENIZER, ESM_BACKBONE, ESM_HEAD, ESM_DEVICE) = load_models()


def clean_sequence(raw):
    if not raw:
        return ""
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith(">")]
    return "".join(lines).upper().replace(" ", "")


def esm_predict(seq):
    encoded = ESM_TOKENIZER([seq], padding=True, return_tensors="pt",
                            truncation=True, max_length=200)
    encoded = {k: v.to(ESM_DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        out = ESM_BACKBONE(**encoded)
        last_hidden = out.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logit = ESM_HEAD(pooled)
        return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# PREDICT tab handler
# ---------------------------------------------------------------------------


def predict(sequence):
    seq = clean_sequence(sequence)
    if not seq:
        return "(enter a sequence)", "", "", "", ""
    if len(seq) > MAX_INPUT_LEN:
        return f"Error: sequence is {len(seq)} aa, max is {MAX_INPUT_LEN}", "", "", "", ""
    if not F.validate_alphabet(seq):
        bad = sorted(set(seq) - set(F.CANONICAL_ALPHABET))
        return f"Error: non-canonical residues: {''.join(bad)}", "", "", "", ""

    xgb_md = "_(XGBoost checkpoint missing.)_"
    xgb_features_md = ""
    if XGB_MODEL is not None:
        feats = F.extract_features(seq)
        col_median = 30.0
        X = np.array([[col_median if np.isnan(v) else v for v in
                       (feats[n] for n in F.FEATURE_NAMES)]], dtype=np.float32)
        proba = float(XGB_MODEL.predict_proba(X)[0, 1])
        xgb_md = (f"**XGBoost P(AMP) = {proba:.3f}** "
                  f"(call: **{'AMP' if proba >= 0.5 else 'non-AMP'}**)\n\n"
                  f"_Phase 1 baseline (test AUC 0.864). 428 engineered features._")
        importance = XGB_MODEL.get_booster().get_score(importance_type="gain")
        ordered = sorted(importance.items(), key=lambda kv: -kv[1])[:5]
        rows = ["| feature | gain | value for this sequence |", "|---|---|---|"]
        for fname, gain in ordered:
            try:
                idx = int(fname[1:])
                friendly = F.FEATURE_NAMES[idx]
                rows.append(f"| {friendly} | {gain:.2f} | {feats[friendly]:.3f} |")
            except (ValueError, IndexError, KeyError):
                rows.append(f"| {fname} | {gain:.2f} |  |")
        xgb_features_md = "**Top-5 features by XGBoost gain:**\n\n" + "\n".join(rows)

    cnn_md = "_(CNN checkpoint missing.)_"
    if CNN_MODEL is not None:
        x = torch.from_numpy(onehot_encode(seq)).unsqueeze(0)
        with torch.no_grad():
            logit = CNN_MODEL(x)
            logit_val = float(logit.item())
            proba_raw = 1.0 / (1.0 + np.exp(-logit_val))
        if CALIBRATOR is not None:
            proba_cal = float(CALIBRATOR.predict_proba([[logit_val]])[0, 1])
            cnn_md = (f"**CNN P(AMP) = {proba_cal:.3f}** "
                      f"(call: **{'AMP' if proba_cal >= 0.5 else 'non-AMP'}**)\n\n"
                      f"_Phase 2A Platt-calibrated (test AUC 0.837). "
                      f"Raw sigmoid was {proba_raw:.3f}._")
        else:
            cnn_md = (f"**CNN P(AMP) = {proba_raw:.3f}** "
                      f"(call at t={CNN_RAW_THRESHOLD:.3f}: "
                      f"**{'AMP' if proba_raw >= CNN_RAW_THRESHOLD else 'non-AMP'}**)")

    esm_md = "_(ESM2 head missing — run `dev.bat esm`.)_"
    if ESM_HEAD is not None:
        proba_esm = esm_predict(seq)
        esm_md = (f"**ESM2-650M head P(AMP) = {proba_esm:.3f}** "
                  f"(call: **{'AMP' if proba_esm >= 0.5 else 'non-AMP'}**)\n\n"
                  f"_Phase 2B current champion (test AUC 0.919, MCC 0.697). "
                  f"Frozen ESM2-650M embeddings + small MLP head._")

    pi = F.physicochemical(seq).get("isoelectric_point", float("nan"))
    summary_md = (f"**Sequence:** `{seq}`  \n**Length:** {len(seq)} aa  \n"
                  f"**Net charge at pH 7:** {F.net_charge_at_ph7(seq):+.2f}  \n"
                  f"**Isoelectric point:** {pi:.2f}  \n"
                  f"**Mean Kyte-Doolittle:** {F.kyte_doolittle_mean(seq):+.2f}  \n"
                  f"**Max Eisenberg moment (11-aa window):** "
                  f"{F.eisenberg_moment_max(seq):.2f}")
    return summary_md, xgb_md, cnn_md, esm_md, xgb_features_md


# ---------------------------------------------------------------------------
# DESIGN tab handler
# ---------------------------------------------------------------------------


def _diff_html(parent, child):
    """Return HTML with mutated residues in <b><span>...</span></b>."""
    out = []
    for i, ch in enumerate(child):
        if i < len(parent) and ch != parent[i]:
            out.append(f"<b><span style='color:#c0392b'>{ch}</span></b>")
        else:
            out.append(ch)
    return "<code>" + "".join(out) + "</code>"


def design(parent_choice, custom_parent, n_mutations, top_k):
    if ESM_HEAD is None:
        return ("_The Design tab needs the ESM2 head loaded. "
                "Run `dev.bat esm` then restart the demo._")
    if parent_choice == "(Custom — paste sequence below)":
        parent = clean_sequence(custom_parent)
        if not parent:
            return "_Paste a custom parent sequence in the textbox below, or pick a library parent._"
        parent_label = "Custom"
    else:
        parent = PARENT_LIBRARY[parent_choice]
        parent_label = parent_choice

    if len(parent) > MAX_INPUT_LEN:
        return f"_Parent is {len(parent)} aa; max supported is {MAX_INPUT_LEN}._"
    if not F.validate_alphabet(parent):
        bad = sorted(set(parent) - set(F.CANONICAL_ALPHABET))
        return f"_Parent contains non-canonical residues: {''.join(bad)}._"

    beam_width = max(int(top_k), 8)
    mutants, parent_score = design_mutants(
        parent, ESM_TOKENIZER, ESM_BACKBONE, ESM_HEAD, ESM_DEVICE,
        n_mutations=int(n_mutations), beam_width=beam_width, top_k=int(top_k),
    )

    header = (f"**Parent ({parent_label}):** {_diff_html(parent, parent)}  \n"
              f"**Parent ESM-head P(AMP) = {parent_score:.4f}**\n\n"
              f"**Top {len(mutants)} mutants after {int(n_mutations)} "
              f"mutation hop(s), beam width {beam_width}:**\n\n")
    rows = ["| rank | mutations | ΔP(AMP) | P(AMP) | sequence |",
            "|---|---|---|---|---|"]
    for i, m in enumerate(mutants, 1):
        delta_str = f"{m.delta:+.4f}"
        rows.append(
            f"| {i} | {m.diff_string() or '(none)'} | "
            f"{delta_str} | {m.score:.4f} | {_diff_html(parent, m.sequence)} |"
        )
    footer = (
        "\n\n_The greedy beam search expands each beam member by every "
        "single-residue substitution, scores all candidates with the ESM2 "
        "head, keeps the top beam-width per hop, and surfaces the global "
        "top-K at the end. Search is deterministic given the same parent / "
        "n_mutations / beam_width. **This is in-silico design — wet-lab "
        "validation is required before any application claim.**_"
    )
    return header + "\n".join(rows) + footer


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


EXAMPLES = [
    ["LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"],
    ["GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSY"],
    ["GIGKFLHSAKKAGKAFVGEIMNS"],
]


def title_suffix():
    parts = []
    if ESM_HEAD is not None: parts.append("ESM2-650M")
    if CNN_MODEL is not None and CALIBRATOR is not None: parts.append("CNN (Platt)")
    elif CNN_MODEL is not None: parts.append("CNN raw")
    if XGB_MODEL is not None: parts.append("XGBoost")
    return " + ".join(parts) if parts else "no models loaded"


PARENT_DROPDOWN = list(PARENT_LIBRARY.keys()) + ["(Custom — paste sequence below)"]


with gr.Blocks(title="AMP Classifier") as demo:
    gr.Markdown(
        f"# AMP Classifier — {title_suffix()}\n\n"
        "Predict whether a peptide is antimicrobial, or design improved "
        "mutants of a known AMP. Cluster-aware 40%-identity test split "
        "throughout."
    )

    with gr.Tabs():
        with gr.Tab("Predict"):
            gr.Markdown(
                "Three models score in parallel: an **XGBoost** trained on "
                "428 engineered features (Phase 1, AUC 0.864), a "
                "Platt-calibrated **CNN** on one-hot sequences (Phase 2A, "
                "AUC 0.837), and an **ESM2-650M + MLP head** on frozen "
                "protein-LM embeddings (Phase 2B champion, AUC 0.919). "
                "Try the cytochrome-c example — XGBoost gets fooled by "
                "the +6 net charge, ESM correctly calls non-AMP."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    inp = gr.Textbox(label="Peptide sequence",
                                     placeholder="e.g. GIGKFLHSAKKFGKAFVGEIMNS",
                                     lines=3)
                    btn = gr.Button("Predict", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=inp)
                with gr.Column(scale=3):
                    summary = gr.Markdown(label="Sequence summary")
            with gr.Row():
                xgb_out = gr.Markdown(label="XGBoost")
                cnn_out = gr.Markdown(label="CNN")
                esm_out = gr.Markdown(label="ESM2-650M (champion)")
            feat_out = gr.Markdown(label="XGBoost feature attribution")
            btn.click(predict, inputs=inp,
                      outputs=[summary, xgb_out, cnn_out, esm_out, feat_out])
            inp.submit(predict, inputs=inp,
                       outputs=[summary, xgb_out, cnn_out, esm_out, feat_out])

        with gr.Tab("Design (Phase 2D)"):
            gr.Markdown(
                "**Discriminator-in-the-loop peptide design.** Pick a "
                "well-known parent AMP (or paste your own), set a mutation "
                "budget, and the ESM2-head champion ranks all single-"
                "residue substitution neighbors. After `n` hops the top-K "
                "highest-scoring mutants are surfaced.\n\n"
                "The search is greedy beam search — fast (a few seconds "
                "per hop on a 4060), deterministic, and complete over "
                "the single-mutation neighborhood it visits. Mutated "
                "residues are highlighted in the result table."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    parent_drop = gr.Dropdown(
                        choices=PARENT_DROPDOWN, value=PARENT_DROPDOWN[1],
                        label="Parent peptide",
                    )
                    custom_parent = gr.Textbox(
                        label="Custom parent (used only if dropdown set to Custom)",
                        placeholder="e.g. GIGKFLHSAKKFGKAFVGEIMNS",
                        lines=2,
                    )
                    n_muts = gr.Slider(1, 5, value=2, step=1,
                                        label="Mutation budget (hops)")
                    top_k_slider = gr.Slider(5, 20, value=10, step=1,
                                              label="Top-K mutants to surface")
                    design_btn = gr.Button("Design", variant="primary")
                with gr.Column(scale=3):
                    design_out = gr.Markdown(label="Top mutants")
            design_btn.click(
                design,
                inputs=[parent_drop, custom_parent, n_muts, top_k_slider],
                outputs=[design_out],
            )

    gr.Markdown(
        "\n---\nFull bootstrap-CI'd reports in `docs/baseline_results.md`, "
        "`docs/cnn_results.md`, `docs/esm_results.md`, "
        "`docs/multiclass_results.md`. Technical writeup in "
        "`report/technical_report.md`."
    )


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port)
