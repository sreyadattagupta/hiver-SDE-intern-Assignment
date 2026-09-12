"""
Fine-tune DistilBERT for Uber intent classification (Phase 6 — REAL supervised fine-tuning).

This performs actual supervised fine-tuning of `distilbert-base-uncased` (encoder + a new 7-way
classification head) on advanced/data/train.csv, then evaluates on the HELD-OUT golden set
(data/golden_set.csv) and compares macro-F1 to the rule baseline on the SAME golden set — the
non-negotiable honesty check from CLAUDE.md (weak-labeled training can distill the rules, so the
only fair verdict is golden-set F1 vs rules).

Outputs:
  advanced/models/intent_distilbert/     the fine-tuned model + tokenizer
  advanced/finetune_eval_report.json     per-class P/R/F1, macro-F1, confusion, vs-rules delta

Requires transformers + torch (present in this env). Weights come from the HF hub cache.
Run:  python -m advanced.train_classifier
"""
from __future__ import annotations

import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import _classify_rules, INTENTS          # noqa: E402
from src.preprocess import clean_for_embedding             # noqa: E402

MODEL = "distilbert-base-uncased"
OUT_MODEL = "advanced/models/intent_distilbert"
REPORT = "advanced/finetune_eval_report.json"
L2I = {lab: i for i, lab in enumerate(INTENTS)}
I2L = {i: lab for lab, i in L2I.items()}


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

    torch.manual_seed(42)
    tr = pd.read_csv("advanced/data/train.csv")
    va = pd.read_csv("advanced/data/val.csv")
    tok = AutoTokenizer.from_pretrained(MODEL)

    def tensors(df):
        enc = tok(df["text"].tolist(), truncation=True, padding="max_length", max_length=64,
                  return_tensors="pt")
        y = torch.tensor([L2I[l] for l in df["label"]], dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    dl_tr = DataLoader(tensors(tr), batch_size=16, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(INTENTS))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # inverse-frequency class weights — without this the tiny safety class (17 ex) is never learned
    counts = tr["label"].value_counts().to_dict()
    w = torch.tensor([len(tr) / (len(INTENTS) * counts.get(lab, 1)) for lab in INTENTS],
                     dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    EPOCHS = 4
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, 0, EPOCHS * len(dl_tr))
    print(f"Fine-tuning DistilBERT on {device} ({EPOCHS} epochs, {len(dl_tr)} steps/epoch, "
          f"class-weighted loss)...")
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for ids, mask, y in dl_tr:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  avg_loss={tot/len(dl_tr):.4f}")

    os.makedirs(OUT_MODEL, exist_ok=True)
    model.save_pretrained(OUT_MODEL)
    tok.save_pretrained(OUT_MODEL)
    print(f"Saved model -> {OUT_MODEL}")

    # ---- evaluate on the held-out golden set ----
    gold = pd.read_csv("data/golden_set.csv")
    texts = [clean_for_embedding(m) for m in gold["customer_msg"]]
    y_true = gold["true_intent"].tolist()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(texts[i:i+32], truncation=True, padding=True, max_length=64, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend([I2L[int(x)] for x in logits.argmax(-1)])

    rules_pred = [_classify_rules(m)["intent"] for m in gold["customer_msg"]]
    ft_macro = f1_score(y_true, preds, average="macro", labels=INTENTS, zero_division=0)
    rules_macro = f1_score(y_true, rules_pred, average="macro", labels=INTENTS, zero_division=0)
    safety_recall = f1_score(y_true, preds, average=None, labels=INTENTS, zero_division=0)[INTENTS.index("safety_incident")]

    report = {
        "model": MODEL,
        "train_size": len(tr), "val_size": len(va), "golden_size": len(gold),
        "finetuned_macro_f1": round(float(ft_macro), 4),
        "finetuned_accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "rules_baseline_macro_f1_same_golden": round(float(rules_macro), 4),
        "delta_vs_rules": round(float(ft_macro - rules_macro), 4),
        "safety_recall": round(float(safety_recall), 4),
        "per_class": classification_report(y_true, preds, labels=INTENTS, zero_division=0, output_dict=True),
        "confusion_matrix_labels": INTENTS,
        "confusion_matrix": confusion_matrix(y_true, preds, labels=INTENTS).tolist(),
    }
    json.dump(report, open(REPORT, "w"), indent=2)
    print("\n" + classification_report(y_true, preds, labels=INTENTS, zero_division=0, digits=3))
    print(f"Fine-tuned macro-F1 : {ft_macro:.3f}")
    print(f"Rules  macro-F1     : {rules_macro:.3f}  (same golden set)")
    print(f"Delta vs rules      : {ft_macro - rules_macro:+.3f}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
