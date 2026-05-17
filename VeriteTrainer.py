import heapq, os, re, json, math, random, logging, warnings, time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModel, set_seed as _hf_set_seed, get_cosine_schedule_with_warmup

def set_seed(seed: int) -> None:
    _hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    precision_score, recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from safetensors.torch import save_file as st_save, load_file as st_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    warnings.warn("safetensors not installed; saving as .pth instead")

try:
    from torch.amp import autocast, GradScaler as _GradScaler
    _AMP_DEVICE = "cuda"
    def _make_scaler(enabled: bool, init_scale: int = 2**11):
        return _GradScaler(_AMP_DEVICE, enabled=enabled, init_scale=init_scale)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler as _GradScaler
    _AMP_DEVICE = "cuda"
    def _make_scaler(enabled: bool, init_scale: int = 2**11):
        return _GradScaler(enabled=enabled, init_scale=init_scale)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    # data
    data_dir:   str  = "/kaggle/input/difraud"
    output_dir: str  = "/kaggle/working"
    model_stem: str  = "model"
    use_liar:   bool = True

    # encoder — ModernBERT-base: 149M params, 2-4× faster than DeBERTa on T4, SDPA-native
    # fallback: "microsoft/deberta-v3-base" (~91 min/epoch) or "microsoft/deberta-v3-large" (SOTA)
    semantic_model_id:     str = "answerdotai/ModernBERT-base"
    freeze_encoder_layers: int = 0

    # dimensions (semantic_dim auto-detected from encoder at model build time)
    semantic_dim: int   = 768    # overridden at init via encoder.config.hidden_size
    ling_dim:     int   = 8      # 7 linguistic + 1 GPT-2 perplexity (0.0 if disabled)
    hidden_dim:   int   = 512
    dropout:      float = 0.1

    # training
    max_seq_len:     int   = 256     # 320 OOMs on 2×T4 with deberta-large; 256 is safe
    batch_size:      int   = 8       # per-GPU effective = 4 with DataParallel; effective batch = 8*2*4=64
    num_epochs:      int   = 5
    lr_encoder:      float = 1e-5
    lr_head:         float = 1e-4
    weight_decay:    float = 0.02
    llrd_decay:      float = 0.9
    warmup_pct:      float = 0.08
    grad_clip:       float = 0.7
    label_smoothing: float = 0.05
    fp16:            bool  = True
    accum_steps:     int   = 4       # doubled to keep effective batch = 64

    # focal loss (gamma=0 → weighted CE)
    focal_gamma: float = 2.0

    # misc
    seed:           int  = 42
    num_workers:    int  = 4
    top_k_evidence: int  = 3     # kept for API compat
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # regularisation
    use_ema:   bool  = True
    ema_decay: float = 0.995
    use_fgm:   bool  = True
    fgm_epsilon: float = 0.25
    use_rdrop:   bool  = False
    rdrop_coef:  float = 0.05
    use_dann:    bool  = False
    dann_lambda:       float = 0.5
    dann_lambda_start: float = 0.01
    dann_num_domains:  int   = 7
    use_supcon:  bool  = True
    supcon_coef: float = 0.1

    # sampling / ensemble
    use_domain_sampler:       bool  = True
    use_hard_negative_mining: bool  = False  # adds ~20 min/epoch of train inference
    hard_neg_conf:            float = 0.80
    hard_neg_boost:           float = 3.0
    snapshot_top_k:           int   = 2    # top-K checkpoints saved per seed

    # MTL: positive domain prediction auxiliary task (Advacheck-style, not adversarial)
    # Teaches encoder domain-specific features instead of suppressing them (DANN does opposite)
    use_domain_mtl:  bool  = True
    domain_mtl_coef: float = 0.1

    # AWP: perturbs model weights (not just embeddings like FGM) — stronger regularization
    # Computationally expensive; apply only from awp_start_epoch onwards
    use_awp:         bool  = True
    awp_epsilon:     float = 0.001
    awp_lr:          float = 0.001
    awp_start_epoch: int   = 3     # start AWP at epoch 4/5 (0-indexed=3) for final regularization

    # pseudo-labeling
    use_pseudo_label:  bool  = False  # requires a separate unlabeled set; False for paper-valid results
    pseudo_label_conf: float = 0.98

    # external data
    use_external_data: bool  = True
    external_weight:   float = 0.30

    # GPT-2 perplexity feature
    use_perplexity:  bool = False  # ~1h precompute; re-enable if budget allows
    perp_model_id:   str  = "distilgpt2"
    perp_batch_size: int  = 64

    # multi-seed
    multi_seed:      bool  = False            # 3 seeds ≈ 15h on 2xT4; single seed fits ~5h
    multi_seed_list: tuple = (42, 1337, 2025)

    # architecture enhancements
    prototype_head_scale:  float = 20.0  # temperature for HypersphericalHead
    use_local_consistency: bool  = True  # segment-level consistency module
    local_cons_segments:   int   = 4     # number of segments for local consistency
    spectral_top_k:        int   = 8     # top-k FFT magnitudes as features

    # calibration / stopping
    early_stopping_patience:    int  = 2
    use_temperature_scaling:    bool = True
    use_compile:                bool = False
    use_gradient_checkpointing: bool = True   # cuts activation memory ~60%; use_reentrant=False for DataParallel compat


CFG = Config()
set_seed(CFG.seed)
log.info(f"Device: {CFG.device} | fp16: {CFG.fp16}")


# ── Lexicons ───────────────────────────────────────────────────────────────────

HEDGE_WORDS = frozenset({
    "maybe","perhaps","possibly","probably","presumably","apparently",
    "seemingly","supposedly","allegedly","might","could","would","should",
    "may","seems","appears","looks","sounds","feels","think","believe",
    "guess","suppose","imagine","reckon","approximately","roughly","around",
    "about","nearly","almost","somewhat","fairly","rather","quite",
    "relatively","generally","usually","often","sometimes","occasionally",
    "tends","suggest","indicate","imply","hint","wonder","uncertain",
    "unclear","unsure","doubtful","questionable","kind","sort","partly",
    "partially","largely",
})
MODAL_VERBS   = frozenset({"can","could","may","might","must","shall","should","will","would"})
NEGATION_CUES = frozenset({"not","no","never","neither","nor","nothing","nobody","none","hardly","barely","scarcely"})

DOMAIN_MAP: Dict[str, int] = {
    "fake_news":0,"job_scams":1,"phishing":2,
    "political_statements":3,"product_reviews":4,"sms":5,"twitter_rumours":6,
}

def _is_deberta(model_id: str) -> bool:
    return "deberta" in model_id.lower()

def _get_encoder_layers(encoder: nn.Module) -> List[nn.Module]:
    # "layers" last: covers ModernBERT (ModernBertModel.layers direct attribute)
    for attr_path in ("encoder.layer","encoder.layers","transformer.layer","model.layers","layers"):
        mod = encoder
        for part in attr_path.split("."):
            mod = getattr(mod, part, None)
            if mod is None: break
        if mod is not None:
            return list(mod)
    return []

def _get_embedding_layer(encoder: nn.Module) -> Optional[nn.Module]:
    """Return the token embedding layer regardless of architecture (DeBERTa / BERT / ModernBERT)."""
    for path in ("embeddings.word_embeddings", "embeddings.tok_embeddings",
                 "model.embeddings.tok_embeddings"):
        mod = encoder
        for part in path.split("."):
            mod = getattr(mod, part, None)
            if mod is None: break
        if mod is not None:
            return mod
    return None


# ── Text features ──────────────────────────────────────────────────────────────

def compute_ling_features(text: str) -> List[float]:
    words  = text.split()
    n      = max(len(words), 1)
    chars  = max(len(text),  1)
    caps   = sum(1 for w in words if w.isupper() and len(w) > 1) / n
    punct  = sum(1 for c in text if c in "!?.,;:") / chars
    excl   = text.count("!") / chars
    ttr    = len({w.lower() for w in words}) / n
    avg_wl = sum(len(w) for w in words) / n / 10.0
    lower  = [w.lower() for w in words]
    hedge  = sum(1 for t in lower if t in HEDGE_WORDS or t in MODAL_VERBS) / n
    neg    = sum(1 for t in lower if t in NEGATION_CUES) / n
    return [caps, punct, excl, ttr, avg_wl, hedge, neg]   # 7 features; 8th added by Dataset


# ── GPT-2 perplexity ───────────────────────────────────────────────────────────

def _compute_perplexity(
    texts: List[str],
    device: str,
    model_id: str = "gpt2",
    batch_size: int = 64,
    max_length: int = 256,
) -> np.ndarray:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    gpt = GPT2LMHeadModel.from_pretrained(model_id).to(device).eval()

    raw: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="GPT-2 perplexity", leave=False):
        chunk = texts[i : i + batch_size]
        enc   = tok(chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length)
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = gpt(input_ids=ids, attention_mask=mask).logits.float()
        shift_log  = logits[:, :-1, :]
        shift_ids  = ids[:, 1:]
        shift_mask = mask[:, 1:].float()
        B, T, V = shift_log.shape
        ce  = F.cross_entropy(shift_log.reshape(-1, V), shift_ids.reshape(-1), reduction="none").view(B, T)
        nll = (ce * shift_mask).sum(1) / shift_mask.sum(1).clamp(min=1)
        raw.extend(nll.cpu().tolist())

    del gpt
    if device == "cuda":
        torch.cuda.empty_cache()

    arr = np.array(raw, dtype=np.float32)
    p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    arr = np.clip(arr, p5, p95)
    arr = (arr - p5) / max(p95 - p5, 1e-6)
    return arr


# ── Data loading ───────────────────────────────────────────────────────────────

def _normalize_label(val) -> Optional[int]:
    if isinstance(val, (int, float)):
        return int(val) if int(val) in (0, 1) else None
    s = str(val).lower().strip()
    if s in ("1","deceptive","lie","false","d","spam","fake"):
        return 1
    if s in ("0","truthful","truth","true","t","ham","real"):
        return 0
    return None


def load_difraud(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _DOMAINS = ["fake_news","job_scams","phishing","political_statements","product_reviews","sms","twitter_rumours"]
    try:
        from huggingface_hub import hf_hub_download
        buckets: Dict[str, List[pd.DataFrame]] = {"train":[],"validation":[],"test":[]}
        for domain in tqdm(_DOMAINS, desc="Downloading DIFrauD"):
            for split in ("train","validation","test"):
                path = hf_hub_download(repo_id="difraud/difraud",
                                       filename=f"{domain}/{split}.jsonl", repo_type="dataset")
                df = pd.read_json(path, lines=True)
                df["domain"] = domain
                buckets[split].append(df)
        train_df = pd.concat(buckets["train"],      ignore_index=True)
        val_df   = pd.concat(buckets["validation"], ignore_index=True)
        test_df  = pd.concat(buckets["test"],       ignore_index=True)
        log.info(f"DIFrauD — train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        return train_df, val_df, test_df
    except Exception as e:
        log.warning(f"hf_hub_download failed: {e}")

    root = Path(data_dir)
    splits: Dict[str, pd.DataFrame] = {}
    for name in ("train","validation","test"):
        p = root / f"{name}.jsonl"
        if p.exists():
            splits[name] = pd.read_json(p, lines=True)
    if len(splits) == 3:
        log.info(f"Loaded from local jsonl in {data_dir}")
        return splits["train"], splits["validation"], splits["test"]

    try:
        df = _load_any(data_dir)
        tr, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
        va, te   = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=42)
        return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Could not load DIFrauD.\n"
        f"  Enable internet: Settings → Internet → ON, then !pip install -q huggingface_hub\n"
        f"  Or place train/validation/test.jsonl in: {data_dir}"
    )


def load_liar() -> pd.DataFrame:
    INT_MAP = {0:1, 1:None, 2:0, 3:0, 4:1, 5:1}
    STR_MAP = {"false":1,"pants-fire":1,"barely-true":1,"mostly-true":0,"true":0,"half-true":None}
    try:
        from datasets import load_dataset
        ds = load_dataset("liar", trust_remote_code=True)
        records = []
        for split_name in ("train","validation","test"):
            if split_name not in ds: continue
            for ex in ds[split_name]:
                raw = ex.get("label", ex.get("Label", None))
                lbl = INT_MAP.get(raw) if isinstance(raw, int) else STR_MAP.get(str(raw).lower().strip())
                if lbl is None: continue
                text = ex.get("statement", ex.get("Statement","")).strip()
                if text:
                    records.append({"text":text,"label":lbl,"domain":"political_statements"})
        df = pd.DataFrame(records).reset_index(drop=True)
        log.info(f"LIAR — {len(df)} samples (deceptive={int(df['label'].sum())})")
        return df
    except Exception as e:
        log.warning(f"Could not load LIAR: {e}")
        return pd.DataFrame(columns=["text","label","domain"])


def load_external_datasets(cfg: "Config") -> pd.DataFrame:
    try:
        from datasets import load_dataset as _ld
    except ImportError:
        log.warning("datasets library not available — skipping external data")
        return pd.DataFrame(columns=["text","label","domain"])

    records: List[Dict] = []

    # SMS spam (ham=truthful=0, spam=deceptive=1)
    try:
        ds = _ld("ucirvine/sms_spam", split="train", trust_remote_code=True)
        n0 = len(records)
        for ex in ds:
            text = str(ex.get("sms","")).strip()
            raw  = ex.get("label", -1)
            lbl  = int(raw) if isinstance(raw, int) and raw in (0,1) else (
                   1 if str(raw).lower()=="spam" else 0 if str(raw).lower()=="ham" else -1)
            if text and lbl in (0,1):
                records.append({"text":text,"label":lbl,"domain":"sms"})
        log.info(f"External SMS spam: {len(records)-n0} samples")
    except Exception as e:
        log.warning(f"SMS spam load failed: {e}")

    # Fake news — GonzaloA: label 0=fake(deceptive), 1=real(truthful) → flip
    try:
        ds = _ld("GonzaloA/fake_news", trust_remote_code=True)
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        n0 = len(records)
        for ex in split:
            text = str(ex.get("text", ex.get("article",""))).strip()
            raw  = ex.get("label", -1)
            if not isinstance(raw, int) or raw not in (0,1): continue
            lbl  = 1 - int(raw)          # flip: 0=fake→1=deceptive, 1=real→0=truthful
            if text and len(text) > 50:
                records.append({"text":text[:1500],"label":lbl,"domain":"fake_news"})
        log.info(f"External fake news: {len(records)-n0} samples")
    except Exception as e:
        log.warning(f"Fake news load failed: {e}")

    if not records:
        log.warning("No external data loaded — continuing without")
        return pd.DataFrame(columns=["text","label","domain"])

    df = pd.DataFrame(records).reset_index(drop=True)
    log.info(f"External total: {len(df)} (deceptive={int(df['label'].sum())})")
    return df


def _ensure_domain_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "domain" not in df.columns:
        df["domain"] = "fake_news"
    df["domain_id"] = df["domain"].map(DOMAIN_MAP).fillna(3).astype(int)
    return df


def load_combined(cfg: "Config") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = load_difraud(cfg.data_dir)

    if cfg.use_liar:
        liar_df = load_liar()
        if len(liar_df) > 0:
            train_df = pd.concat([train_df, liar_df], ignore_index=True)

    if cfg.use_external_data:
        ext_df = load_external_datasets(cfg)
        if len(ext_df) > 0:
            n_main   = len(train_df)
            target_n = int(n_main * cfg.external_weight / max(1.0 - cfg.external_weight, 1e-6))
            if len(ext_df) > target_n:
                ext_df = ext_df.sample(n=target_n, random_state=cfg.seed)
            train_df = pd.concat([train_df, ext_df], ignore_index=True)
            log.info(f"After external data: train={len(train_df)}")

    train_df = _ensure_domain_ids(train_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True))
    val_df   = _ensure_domain_ids(val_df)
    test_df  = _ensure_domain_ids(test_df)
    log.info(f"Combined — train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return train_df, val_df, test_df


def _load_any(data_dir: str) -> pd.DataFrame:
    root = Path(data_dir)
    for csv_path in sorted(root.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        tc = next((c for c in df.columns if c in ("text","statement","transcript","content")), None)
        lc = next((c for c in df.columns if c in ("label","verdict","class","deceptive")), None)
        if tc and lc:
            df = df[[tc,lc]].rename(columns={tc:"text",lc:"label"})
            df["label"] = df["label"].apply(_normalize_label)
            df = df.dropna(subset=["text","label"]).copy()
            df["label"] = df["label"].astype(int)
            return df.reset_index(drop=True)
    records = []
    for label_name, lv in [("Deceptive",1),("deceptive",1),("Truthful",0),("truthful",0)]:
        subdir = root / label_name
        if subdir.is_dir():
            for f in sorted(subdir.glob("*.txt")):
                txt = f.read_text(encoding="utf-8", errors="ignore").strip()
                if txt: records.append({"text":txt,"label":lv})
    if records:
        return pd.DataFrame(records).reset_index(drop=True)
    raise FileNotFoundError(f"No recognizable data in {data_dir}.")


# ── Dataset ────────────────────────────────────────────────────────────────────

class DeceptionDataset(Dataset):
    def __init__(
        self,
        texts:       List[str],
        labels:      List[int],
        tokenizer,
        cfg:         Config,
        domain_ids:  Optional[np.ndarray] = None,
        perp_scores: Optional[np.ndarray] = None,
    ):
        self.texts       = texts
        self.labels      = labels
        self.tokenizer   = tokenizer
        self.cfg         = cfg
        self.domain_ids  = domain_ids
        self.perp_scores = perp_scores

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        text  = self.texts[idx]
        enc   = self.tokenizer(text, max_length=self.cfg.max_seq_len,
                               truncation=True, padding="max_length", return_tensors="pt")
        ling  = compute_ling_features(text)
        ling.append(float(self.perp_scores[idx]) if self.perp_scores is not None else 0.0)
        did   = int(self.domain_ids[idx]) if self.domain_ids is not None else 3
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ling_feats":     torch.tensor(ling, dtype=torch.float),
            "domain_id":      torch.tensor(did, dtype=torch.long),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Model modules ──────────────────────────────────────────────────────────────

class ProjectionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: Optional[torch.Tensor], label_smoothing: float):
        super().__init__()
        self.gamma = gamma; self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lf = logits.float()
        ce = F.cross_entropy(lf, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        fw = (1.0 - pt) ** self.gamma
        if self.alpha is not None: fw = fw * self.alpha.float()[targets]
        return (fw * ce).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B    = features.shape[0]
        feat = F.normalize(features.float(), dim=-1)
        sim  = (feat @ feat.T) / self.temperature
        eye  = torch.eye(B, dtype=torch.bool, device=features.device)
        sim  = sim.masked_fill(eye, -1e4)
        pos  = (labels.view(-1,1) == labels.view(1,-1)) & ~eye
        if not pos.any(): return feat.new_zeros(1).squeeze()
        lp  = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        per = -(lp * pos.float()).sum(1) / pos.float().sum(1).clamp(min=1)
        return per[pos.any(1)].mean()


class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lam=1.0): super().__init__(); self.lam = lam
    def forward(self, x): return _GRLFunction.apply(x, self.lam)

class DomainAdversarialHead(nn.Module):
    def __init__(self, hd, nd, drop):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(hd, hd//2), nn.GELU(), nn.Dropout(drop), nn.Linear(hd//2, nd))
    def forward(self, x): return self.clf(x)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model; self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
        self._backup: Dict = {}
    @torch.inference_mode()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)
    def apply(self):
        self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        cast = {k: s.to(dtype=self._backup[k].dtype, device=self._backup[k].device)
                for k, s in self.shadow.items()}
        self.model.load_state_dict(cast, strict=False)
    def restore(self):
        self.model.load_state_dict(self._backup)


class FGM:
    def __init__(self, model: nn.Module, epsilon: float = 0.25):
        self.model = model; self.epsilon = epsilon; self.backup: Dict = {}
    def attack(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm > 0: param.data.add_(self.epsilon * param.grad / norm)
    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


class AWP:
    """Adversarial Weight Perturbation: perturbs all weight matrices (not just embeddings).
    Strictly stronger than FGM for fine-tuning. Apply after unscale_, from awp_start_epoch."""
    def __init__(self, model: nn.Module, epsilon: float = 0.001, lr: float = 0.001):
        self.model   = model
        self.epsilon = epsilon
        self.lr      = lr
        self.backup:     Dict[str, torch.Tensor] = {}
        self.backup_eps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "weight" in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    g_eps = self.epsilon * param.abs().detach()
                    self.backup_eps[name] = (self.backup[name] - g_eps,
                                             self.backup[name] + g_eps)
                norm_g = param.grad.float().norm()
                norm_w = param.data.float().norm()
                if norm_g > 0 and norm_w > 0 and norm_g.isfinite() and norm_w.isfinite():
                    r = self.lr * param.grad.float() / (norm_g + 1e-6) * (norm_w + 1e-6)
                    param.data.add_(r.to(param.dtype))
                    lo, hi = self.backup_eps[name]
                    param.data.clamp_(lo, hi)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()
        self.backup_eps.clear()


# ── Architectural enhancements ────────────────────────────────────────────────

def _spectral_features(hidden: torch.Tensor, mask: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Frequency-domain features: top-k FFT magnitudes + spectral centroid + entropy → (B, top_k+2)."""
    signal  = (hidden.float() * mask.unsqueeze(-1).float()).mean(-1)          # (B, L)
    fft_mag = torch.fft.rfft(signal, dim=-1).abs()                            # (B, F)
    norm    = fft_mag.sum(-1, keepdim=True) + 1e-8
    top_mags = fft_mag[:, 1 : top_k + 1] / norm                               # skip DC, (B, top_k)
    freqs    = torch.arange(fft_mag.size(-1), device=fft_mag.device).float()
    centroid = ((fft_mag * freqs).sum(-1) / norm.squeeze(-1) / (fft_mag.size(-1) + 1e-8)).unsqueeze(-1)
    prob     = fft_mag / norm
    entropy  = -(prob * (prob + 1e-8).log()).sum(-1, keepdim=True)
    return torch.cat([top_mags, centroid, entropy], dim=-1)                   # (B, top_k+2)


class LocalConsistencyModule(nn.Module):
    """Splits token sequence into n_segs chunks; cross-segment attention scores contradiction.
    Learnable Q/K projections capture incompatibilities beyond raw cosine similarity."""
    def __init__(self, in_dim: int, n_segs: int = 4, out_dim: int = 256):
        super().__init__()
        self.n_segs   = n_segs
        self.seg_proj = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.LayerNorm(out_dim))
        self.q_proj   = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj   = nn.Linear(out_dim, out_dim, bias=False)
        self.scale    = out_dim ** -0.5

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, H = hidden.shape
        seg_len = max(1, L // self.n_segs)
        m       = mask.unsqueeze(-1).float()
        segs = []
        for s in range(self.n_segs):
            h_s  = hidden[:, s * seg_len : (s + 1) * seg_len]
            m_s  = m[:, s * seg_len : (s + 1) * seg_len]
            pool = (h_s * m_s).sum(1) / m_s.sum(1).clamp(min=1)
            segs.append(self.seg_proj(pool))
        segs = torch.stack(segs, dim=1)                                        # (B, n_segs, out_dim)
        Q    = self.q_proj(segs)
        K    = self.k_proj(segs)
        sim  = torch.bmm(Q, K.transpose(1, 2)) * self.scale                   # (B, n_segs, n_segs)
        n_off = self.n_segs * (self.n_segs - 1)
        cons  = (sim.sum(-1).sum(-1) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)) / n_off
        ctx   = torch.bmm(torch.softmax(sim, dim=-1), segs)                   # (B, n_segs, out_dim)
        return ctx.mean(1), cons                                               # (B, out_dim), (B,)


class AttentionPooling(nn.Module):
    """Learned scalar attention over token positions — better than mean for sparse deception signals."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(hidden).squeeze(-1)                                # (B, L)
        scores  = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)                                # (B, L)
        return (hidden * weights.unsqueeze(-1)).sum(1)                         # (B, H)


class HypersphericalHead(nn.Module):
    """Prototype logits + residual linear: geometric robustness + local flexibility."""
    def __init__(self, in_dim: int, n_classes: int = 2, scale: float = 20.0):
        super().__init__()
        self.scale      = scale
        self.prototypes = nn.Parameter(F.normalize(torch.randn(n_classes, in_dim), dim=-1))
        self.linear     = nn.Linear(in_dim, n_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proto  = self.scale * F.normalize(x, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        return proto + self.linear(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class DeceptionReasoningModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        _kw = {} if _is_deberta(cfg.semantic_model_id) else {"attn_implementation": "sdpa"}
        try:
            self.semantic_encoder = AutoModel.from_pretrained(cfg.semantic_model_id, **_kw)
        except Exception as e:
            log.warning(f"attn_implementation='sdpa' failed ({e}), falling back")
            self.semantic_encoder = AutoModel.from_pretrained(cfg.semantic_model_id)
        # ModernBERT calls _maybe_set_compile() and self.dtype inside forward(), both of
        # which use next(... for param in self.parameters() ...) without a default.
        # On DataParallel replicas the generator is exhausted → StopIteration crash.
        # Fix 1: patch _maybe_set_compile to a no-op.
        if hasattr(self.semantic_encoder, "_maybe_set_compile"):
            self.semantic_encoder._maybe_set_compile = lambda: None
        # Fix 2: patch the concrete encoder class's `dtype` property to use next(..., fallback).
        # This shadows PreTrainedModel.dtype only for this concrete class (e.g. ModernBertModel).
        if not _is_deberta(cfg.semantic_model_id):
            _enc_cls = type(self.semantic_encoder)
            if not isinstance(_enc_cls.__dict__.get("dtype"), property):
                import torch as _torch
                _enc_cls.dtype = property(
                    lambda self: next(
                        (p.dtype for p in self.parameters() if p.is_floating_point()),
                        _torch.float32,
                    )
                )
        self._freeze_bottom_layers(self.semantic_encoder, cfg.freeze_encoder_layers)

        sem_dim = getattr(self.semantic_encoder.config, "hidden_size", cfg.semantic_dim)
        self.semantic_proj = ProjectionLayer(sem_dim,      cfg.hidden_dim, cfg.dropout)
        self.ling_proj     = ProjectionLayer(cfg.ling_dim, cfg.hidden_dim, cfg.dropout)
        self.attn_pool     = AttentionPooling(sem_dim)

        _cons_dim = cfg.hidden_dim // 2 if cfg.use_local_consistency else 0
        self.local_cons = (
            LocalConsistencyModule(sem_dim, cfg.local_cons_segments, _cons_dim)
            if cfg.use_local_consistency else None
        )

        fused_dim = cfg.hidden_dim * 2 + _cons_dim + cfg.spectral_top_k + 2  # top_k + centroid + entropy
        self.pre_feat = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, cfg.hidden_dim), nn.GELU(),
        )
        self.dropouts    = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(5)])
        self.head        = HypersphericalHead(cfg.hidden_dim, n_classes=2, scale=cfg.prototype_head_scale)
        self.grl         = GradientReversalLayer(cfg.dann_lambda)
        self.domain_head = DomainAdversarialHead(cfg.hidden_dim, cfg.dann_num_domains, cfg.dropout)
        self.register_buffer("temperature_scale", torch.ones(1))

    @staticmethod
    def _freeze_bottom_layers(encoder, n):
        if n <= 0: return
        if hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters(): p.requires_grad = False
        for layer in _get_encoder_layers(encoder)[:n]:
            for p in layer.parameters(): p.requires_grad = False

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, ling_feats=None) -> Dict[str, torch.Tensor]:
        _dev = input_ids.device.type
        # Encoder always fp32: fp16 SDPA on T4 (sm75) can overflow mid-training → NaN weights
        with torch.amp.autocast(device_type=_dev, enabled=False):
            sem_out = self.semantic_encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden  = sem_out.last_hidden_state.float()
        sem_emb = self.semantic_proj(self.attn_pool(hidden, attention_mask))

        if ling_feats is not None:
            feat_emb = self.ling_proj(ling_feats.to(sem_emb.device))
        else:
            feat_emb = sem_emb.new_zeros(sem_emb.size(0), self.cfg.hidden_dim)

        parts = [sem_emb, feat_emb]
        if self.local_cons is not None:
            cons_emb, _ = self.local_cons(hidden, attention_mask)
            parts.append(cons_emb)
        parts.append(_spectral_features(hidden, attention_mask, self.cfg.spectral_top_k))

        fused = torch.cat(parts, dim=-1)
        feat  = self.pre_feat(fused)
        if self.training:
            logits = torch.stack([self.head(d(feat)) for d in self.dropouts]).mean(0)
        else:
            logits = self.head(feat)
        logits = logits / self.temperature_scale.clamp(min=0.1)
        domain_logits = (
            self.domain_head(self.grl(sem_emb)) if self.cfg.use_dann          # adversarial
            else self.domain_head(sem_emb) if self.cfg.use_domain_mtl          # positive MTL
            else None
        )
        return {"logits": logits, "sem_emb": sem_emb, "domain_logits": domain_logits}


# ── Optimizer (LLRD) ───────────────────────────────────────────────────────────

def _build_optimizer(raw_model: DeceptionReasoningModel, cfg: Config) -> AdamW:
    no_decay  = {"bias","LayerNorm.weight","layer_norm.weight"}
    encoder   = raw_model.semantic_encoder
    enc_layers = _get_encoder_layers(encoder)
    n = len(enc_layers)
    groups: List[Dict] = []

    def _add(module, base_lr):
        wd, no_wd = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad: continue
            (no_wd if any(nd in name for nd in no_decay) else wd).append(param)
        if wd:    groups.append({"params":wd,    "lr":base_lr, "weight_decay":cfg.weight_decay})
        if no_wd: groups.append({"params":no_wd, "lr":base_lr, "weight_decay":0.0})

    if hasattr(encoder, "embeddings"):
        _add(encoder.embeddings, cfg.lr_encoder * (cfg.llrd_decay ** n))
    for i, layer in enumerate(enc_layers):
        _add(layer, cfg.lr_encoder * (cfg.llrd_decay ** (n - i - 1)))

    enc_ids = {id(p) for p in encoder.parameters()}
    hw, hn  = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad or id(param) in enc_ids: continue
        (hn if any(nd in name for nd in no_decay) else hw).append(param)
    if hw: groups.append({"params":hw, "lr":cfg.lr_head, "weight_decay":cfg.weight_decay})
    if hn: groups.append({"params":hn, "lr":cfg.lr_head, "weight_decay":0.0})
    return AdamW(groups)


def _move_batch(batch: Dict, device: str) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ── Hard negative mining ───────────────────────────────────────────────────────

def _update_sampler_weights(
    model: nn.Module,
    plain_loader: DataLoader,
    base_weights: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(plain_loader, desc="hard-neg scan", leave=False):
            batch = _move_batch(batch, cfg.device)
            out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          ling_feats=batch.get("ling_feats"))
            p = torch.softmax(out["logits"].float(), -1)[:, 1]
            all_probs.extend(p.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
    model.train()

    new_w = base_weights.clone()
    n_hard = 0
    for i, (prob, label) in enumerate(zip(all_probs, all_labels)):
        pred = 1 if prob >= 0.5 else 0
        if pred != label and max(prob, 1.0 - prob) > cfg.hard_neg_conf:
            new_w[i] *= cfg.hard_neg_boost
            n_hard += 1
    log.info(f"  Hard neg mining: {n_hard}/{len(all_labels)} samples boosted ×{cfg.hard_neg_boost:.0f}")
    return new_w


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    fold:     int,
    cfg:      Config,
) -> Dict:
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log.info(f"Fold {fold+1} | train={len(train_df)} val={len(val_df)}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.semantic_model_id)
    model     = DeceptionReasoningModel(cfg).to(cfg.device)

    if _is_deberta(cfg.semantic_model_id):
        model.semantic_encoder.float()
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

    n_gpu        = torch.cuda.device_count() if cfg.device == "cuda" else 1
    loader_batch = cfg.batch_size * max(1, n_gpu)
    _pw          = False   # persistent_workers freezes Kaggle notebooks

    _amp_dtype = torch.float16
    if cfg.device == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        if cap >= 8 and not _is_deberta(cfg.semantic_model_id):
            _amp_dtype = torch.bfloat16
        elif cap < 8 and cfg.use_fgm:
            cfg.use_fgm = False
            log.warning("FGM disabled on pre-Ampere GPU")

    perp_col = train_df["perplexity"].values if "perplexity" in train_df.columns else None
    _train_ds = DeceptionDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, cfg,
        domain_ids=train_df["domain_id"].values if "domain_id" in train_df else None,
        perp_scores=perp_col,
    )

    _labels   = train_df["label"].values
    _l_counts = np.bincount(_labels, minlength=2).clip(min=1)
    _w        = torch.tensor(1.0 / _l_counts[_labels], dtype=torch.float)

    def _make_loader(w, shuffle=False):
        if w is not None:
            sampler = torch.utils.data.WeightedRandomSampler(w, len(train_df), replacement=True)
            return DataLoader(_train_ds, batch_size=loader_batch, sampler=sampler,
                              num_workers=cfg.num_workers, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=_pw,
                              prefetch_factor=2 if _pw else None)
        return DataLoader(_train_ds, batch_size=loader_batch, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=collate_fn,
                          pin_memory=True, persistent_workers=_pw,
                          prefetch_factor=2 if _pw else None)

    train_loader = _make_loader(_w if cfg.use_domain_sampler else None,
                                shuffle=not cfg.use_domain_sampler)
    plain_loader = (DataLoader(_train_ds, batch_size=loader_batch * 2, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
                    if cfg.use_hard_negative_mining else None)
    val_loader = DataLoader(
        DeceptionDataset(
            val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, cfg,
            domain_ids=val_df["domain_id"].values if "domain_id" in val_df else None,
            perp_scores=val_df["perplexity"].values if "perplexity" in val_df.columns else None,
        ),
        batch_size=loader_batch * 2, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=_pw, prefetch_factor=2 if _pw else None,
    )

    # Gradient checkpointing — must be enabled BEFORE DataParallel wrapping.
    # use_reentrant=False is required for DataParallel compatibility.
    if cfg.use_gradient_checkpointing and hasattr(model.semantic_encoder, "gradient_checkpointing_enable"):
        try:
            model.semantic_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            log.info("Gradient checkpointing enabled (use_reentrant=False)")
        except TypeError:
            model.semantic_encoder.gradient_checkpointing_enable()
            log.info("Gradient checkpointing enabled (legacy API)")
        except Exception as e:
            log.warning(f"Gradient checkpointing failed: {e}")

    raw_model = model
    if n_gpu > 1:
        log.info(f"DataParallel — {n_gpu} GPUs")
        model = nn.DataParallel(model)
        raw_model = model.module
    elif cfg.use_compile and hasattr(torch, "compile"):
        try: model = torch.compile(model); log.info("torch.compile() enabled")
        except Exception as e: log.warning(f"torch.compile() failed: {e}")

    optimizer = _build_optimizer(raw_model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    total_steps     = steps_per_epoch * cfg.num_epochs
    warmup_steps    = max(1, int(cfg.warmup_pct * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp = cfg.fp16 and cfg.device == "cuda"
    scaler  = _make_scaler(use_amp)

    n_pos   = int(train_df["label"].sum())
    n_neg   = len(train_df) - n_pos
    alpha_t = torch.tensor(
        [len(train_df) / (2 * max(n_neg,1)), len(train_df) / (2 * max(n_pos,1))],
        dtype=torch.float, device=cfg.device,
    )
    criterion = FocalLoss(cfg.focal_gamma, alpha_t, cfg.label_smoothing)
    ema       = EMA(raw_model, cfg.ema_decay) if cfg.use_ema else None
    fgm       = FGM(raw_model, cfg.fgm_epsilon) if cfg.use_fgm else None
    awp       = AWP(raw_model, cfg.awp_epsilon, cfg.awp_lr) if cfg.use_awp else None
    supcon    = SupConLoss(0.07).to(cfg.device) if cfg.use_supcon else None

    best_f1, best_threshold, best_state, patience_counter = 0.0, 0.5, None, 0
    global_step  = 0
    snap_counter = 0
    snapshot_heap: List[Tuple] = []   # (f1, counter, state_dict)
    _nan_streak  = 0   # consecutive non-finite loss steps; triggers EMA recovery at 20

    for epoch in range(cfg.num_epochs):
        if cfg.use_dann:
            _grl_p = global_step / max(total_steps, 1)

        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        _n_batches   = len(train_loader)
        _epoch_start = time.time()
        _ema_loss    = None   # exponential moving average of per-step loss for display
        for i, batch in enumerate(train_loader):
            batch   = _move_batch(batch, cfg.device)
            is_step = (i + 1) % cfg.accum_steps == 0 or (i + 1) == len(train_loader)

            if cfg.use_dann:
                raw_model.grl.lam = (
                    cfg.dann_lambda_start + (cfg.dann_lambda - cfg.dann_lambda_start) *
                    (global_step / max(total_steps, 1))
                )

            with autocast(_AMP_DEVICE, dtype=_amp_dtype, enabled=use_amp):
                out  = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             ling_feats=batch.get("ling_feats"))
                loss = criterion(out["logits"], batch["label"])

                if cfg.use_dann and out["domain_logits"] is not None:
                    loss = loss + F.cross_entropy(out["domain_logits"].float(), batch["domain_id"])

                if cfg.use_domain_mtl and not cfg.use_dann and out["domain_logits"] is not None:
                    loss = loss + cfg.domain_mtl_coef * F.cross_entropy(
                        out["domain_logits"].float(), batch["domain_id"]
                    )

                if supcon is not None:
                    loss = loss + cfg.supcon_coef * supcon(out["sem_emb"], batch["label"])

                if cfg.use_rdrop:
                    out2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 ling_feats=batch.get("ling_feats"))
                    p1 = F.softmax(out["logits"].float(),  dim=-1)
                    p2 = F.softmax(out2["logits"].float(), dim=-1)
                    rdrop = (F.kl_div(p1.log(), p2, reduction="batchmean") +
                             F.kl_div(p2.log(), p1, reduction="batchmean")) / 2.0
                    loss = loss + cfg.rdrop_coef * rdrop

                loss = loss / cfg.accum_steps

            lv = loss.item()
            if not math.isfinite(lv):
                _nan_streak += 1
                if _nan_streak <= 3:
                    log.warning(f"Non-finite loss at step {i+1} — skipping")
                if _nan_streak == 20:
                    # Weights are NaN — recover from EMA shadow (float32, always valid)
                    log.warning("NaN cascade at step %d — restoring EMA weights + resetting optimizer", i+1)
                    if ema is not None:
                        _ema_sd = {k: s.to(dtype=raw_model.state_dict()[k].dtype,
                                           device=raw_model.state_dict()[k].device)
                                   for k, s in ema.shadow.items()}
                        raw_model.load_state_dict(_ema_sd, strict=False)
                    optimizer.state.clear()   # stale Adam moments may also be NaN
                    _nan_streak = 0
                optimizer.zero_grad(set_to_none=True); continue
            _nan_streak = 0

            scaler.scale(loss).backward()
            _step_loss = lv * cfg.accum_steps
            total_loss += _step_loss
            _ema_loss   = _step_loss if _ema_loss is None else 0.98 * _ema_loss + 0.02 * _step_loss
            if (i + 1) % 200 == 0 or (i + 1) == _n_batches:
                _elapsed = time.time() - _epoch_start
                _eta     = _elapsed / (i + 1) * (_n_batches - i - 1)
                log.info(f"  Epoch {epoch+1}/{cfg.num_epochs} | "
                         f"step {i+1}/{_n_batches} | "
                         f"loss(ema)={_ema_loss:.4f} | "
                         f"loss(avg)={total_loss / (i + 1):.4f} | "
                         f"elapsed={_elapsed/60:.1f}m | ETA={_eta/60:.1f}m")

            if is_step:
                if fgm is not None:
                    fgm.attack()
                    with autocast(_AMP_DEVICE, dtype=_amp_dtype, enabled=use_amp):
                        out_adv  = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                        ling_feats=batch.get("ling_feats"))
                        loss_adv = criterion(out_adv["logits"], batch["label"]) / cfg.accum_steps
                    if math.isfinite(loss_adv.item()): scaler.scale(loss_adv).backward()
                    fgm.restore()

                scaler.unscale_(optimizer)

                # AWP: weight perturbation from awp_start_epoch (needs unscaled grads)
                if awp is not None and epoch >= cfg.awp_start_epoch:
                    awp.attack()
                    with autocast(_AMP_DEVICE, dtype=_amp_dtype, enabled=use_amp):
                        out_awp  = model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"],
                                         ling_feats=batch.get("ling_feats"))
                        loss_awp = criterion(out_awp["logits"], batch["label"]) / cfg.accum_steps
                    if math.isfinite(loss_awp.item()):
                        loss_awp.backward()
                    awp.restore()

                gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(gn):
                    log.warning(f"Non-finite grad at step {global_step+1}")
                    # Do NOT call optimizer.step() — bad grads can corrupt Adam state.
                    # scaler.update() alone still reduces the scale factor.
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None: ema.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # ── Validation ────────────────────────────────────────────────────────
        if ema is not None: ema.apply()
        model.eval()
        preds, labels, probs, dids = [], [], [], []
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="validating", leave=False):
                batch = _move_batch(batch, cfg.device)
                out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              ling_feats=batch.get("ling_feats"))
                p = torch.softmax(out["logits"], -1)[:, 1]
                preds.extend(out["logits"].argmax(-1).cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
                probs.extend(p.cpu().tolist())
                dids.extend(batch["domain_id"].cpu().tolist())

        opt_thresh = find_optimal_thresholds(probs, labels, dids, cfg.dann_num_domains)
        preds_opt  = [1 if p >= opt_thresh.get(d, opt_thresh[-1]) else 0
                      for p, d in zip(probs, dids)]
        f1  = f1_score(labels, preds_opt, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds_opt)
        try:   auc = roc_auc_score(labels, probs)
        except: auc = 0.5
        log.info(f"  Epoch {epoch+1:02d}/{cfg.num_epochs} | "
                 f"loss={total_loss/max(1,len(train_loader)):.4f} | "
                 f"F1={f1:.4f} | Acc={acc:.4f} | AUC={auc:.4f}")

        # ── Snapshot save ─────────────────────────────────────────────────────
        snap_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        entry      = (f1, snap_counter, snap_state)
        snap_counter += 1
        if len(snapshot_heap) < cfg.snapshot_top_k:
            heapq.heappush(snapshot_heap, entry)
        elif f1 > snapshot_heap[0][0]:
            heapq.heapreplace(snapshot_heap, entry)

        if f1 > best_f1:
            best_f1 = f1; best_threshold = opt_thresh
            best_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
            patience_counter = 0
            # Save to disk immediately so a crash/timeout doesn't lose the best checkpoint
            _ckpt = os.path.join(cfg.output_dir, f"{cfg.model_stem}_fold{fold}_best.safetensors")
            try:
                if HAS_SAFETENSORS:
                    from safetensors.torch import save_file as _sf
                    _sf({k: v.contiguous() for k, v in best_state.items()}, _ckpt)
                else:
                    _ckpt = _ckpt.replace(".safetensors", ".pth")
                    torch.save(best_state, _ckpt)
                log.info(f"  Checkpoint saved → {_ckpt}  (F1={best_f1:.4f})")
            except Exception as _e:
                log.warning(f"  Checkpoint save failed: {_e}")
        else:
            patience_counter += 1

        if ema is not None: ema.restore()

        # ── Hard negative mining (once, after epoch 0) ─────────────────────
        if cfg.use_hard_negative_mining and epoch == 0 and plain_loader is not None:
            _w = _update_sampler_weights(raw_model, plain_loader, _w, cfg)
            train_loader = _make_loader(_w)

        if patience_counter >= cfg.early_stopping_patience:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    log.info(f"  Fold {fold+1} best val macro-F1: {best_f1:.4f}")
    raw_model.load_state_dict(best_state)
    return {
        "model": raw_model, "f1": best_f1, "state": best_state, "fold": fold,
        "tokenizer": tokenizer, "val_df": val_df, "threshold": best_threshold,
        "snapshots": list(snapshot_heap),
    }


# ── Ensemble inference ─────────────────────────────────────────────────────────

def _ensemble_infer_snapshots(
    model:     DeceptionReasoningModel,
    snapshots: List[Tuple],   # (f1, counter, state_dict)
    df:        pd.DataFrame,
    tokenizer,
    cfg:       Config,
) -> torch.Tensor:
    loader = DataLoader(
        DeceptionDataset(
            df["text"].tolist(), df["label"].tolist(), tokenizer, cfg,
            domain_ids=df["domain_id"].values if "domain_id" in df else None,
            perp_scores=df["perplexity"].values if "perplexity" in df.columns else None,
        ),
        batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    sum_logits: Optional[torch.Tensor] = None
    for _, _cnt, state in snapshots:
        model.load_state_dict(state)
        model.eval()
        fold_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, cfg.device)
                out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              ling_feats=batch.get("ling_feats"))
                fold_logits.append(out["logits"].cpu().float())
        logits = torch.cat(fold_logits)
        sum_logits = logits if sum_logits is None else sum_logits + logits
    return sum_logits / len(snapshots)


# ── Pseudo-labeling ────────────────────────────────────────────────────────────

def pseudo_label_round(
    model:     DeceptionReasoningModel,
    test_df:   pd.DataFrame,
    train_df:  pd.DataFrame,
    cfg:       Config,
    tokenizer,
) -> pd.DataFrame:
    loader = DataLoader(
        DeceptionDataset(
            test_df["text"].tolist(), [0] * len(test_df), tokenizer, cfg,
            domain_ids=test_df["domain_id"].values if "domain_id" in test_df else None,
            perp_scores=test_df["perplexity"].values if "perplexity" in test_df.columns else None,
        ),
        batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    model.eval()
    probs: List[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="pseudo-label", leave=False):
            batch = _move_batch(batch, cfg.device)
            out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          ling_feats=batch.get("ling_feats"))
            probs.extend(torch.softmax(out["logits"].float(), -1)[:, 1].cpu().tolist())

    rows = []
    for i, prob in enumerate(probs):
        if prob >= cfg.pseudo_label_conf:          lbl = 1
        elif prob <= 1.0 - cfg.pseudo_label_conf:  lbl = 0
        else:                                      continue
        row = test_df.iloc[i].to_dict()
        row["label"] = lbl
        rows.append(row)

    if not rows:
        log.info("Pseudo-labeling: no high-confidence predictions found")
        return train_df

    pseudo_df = pd.DataFrame(rows)
    augmented  = pd.concat([train_df, pseudo_df], ignore_index=True)
    augmented  = augmented.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    log.info(f"Pseudo-labeling: +{len(rows)} samples → train={len(augmented)}")
    return augmented


# ── Temperature scaling ────────────────────────────────────────────────────────

def calibrate_temperature(raw_model: DeceptionReasoningModel, val_loader: DataLoader, cfg: Config) -> float:
    raw_model.eval(); raw_model.temperature_scale.fill_(1.0)
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="calibration", leave=False):
            batch = _move_batch(batch, cfg.device)
            out   = raw_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              ling_feats=batch.get("ling_feats"))
            all_logits.append(out["logits"].cpu().float())
            all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    T   = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.1), labels)
        loss.backward(); return loss
    opt.step(closure)
    T_val = float(max(0.1, min(10.0, T.item())))
    log.info(f"Temperature scaling: T={T_val:.3f}")
    raw_model.temperature_scale.fill_(T_val)
    return T_val


# ── Threshold search ──────────────────────────────────────────────────────────

def find_optimal_threshold(probs: List[float], labels: List[int]) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = [1 if p >= t else 0 for p in probs]
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f1: best_f1, best_t = f, float(t)
    return best_t


def find_optimal_thresholds(probs, labels, domain_ids, n_domains) -> Dict[int, float]:
    thresholds = {-1: find_optimal_threshold(probs, labels)}
    for d in range(n_domains):
        idx = [i for i, did in enumerate(domain_ids) if did == d]
        if len(idx) < 10: continue
        dp, dl = [probs[i] for i in idx], [labels[i] for i in idx]
        if len(set(dl)) < 2: continue
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.20, 0.81, 0.01):
            preds = [1 if p >= t else 0 for p in dp]
            f = f1_score(dl, preds, average="macro", zero_division=0)
            if f > best_f1: best_f1, best_t = f, float(t)
        thresholds[d] = best_t
    return thresholds


# ── Save / load ────────────────────────────────────────────────────────────────

def save_model(model: nn.Module, cfg: Config, suffix: str = "", tokenizer=None) -> str:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.model_stem}{suffix}"
    if HAS_SAFETENSORS:
        path = os.path.join(cfg.output_dir, f"{stem}.safetensors")
        st_save({k: v.contiguous() for k, v in model.state_dict().items()}, path)
    else:
        path = os.path.join(cfg.output_dir, f"{stem}.pth")
        torch.save(model.state_dict(), path)
    with open(os.path.join(cfg.output_dir, f"{stem}_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(cfg.output_dir, f"{stem}_tokenizer"))
    log.info(f"Saved → {path}")
    return path


def load_model(path: str, cfg: Config) -> DeceptionReasoningModel:
    model = DeceptionReasoningModel(cfg)
    if path.endswith(".safetensors") and HAS_SAFETENSORS:
        state = st_load(path, device=cfg.device)
    else:
        state = torch.load(path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device).eval()
    log.info(f"Loaded ← {path}")
    return model


# ── Integrated gradients ──────────────────────────────────────────────────────

def integrated_gradients(model, batch, target_class=1, steps=20):
    _dev = model.cfg.device; model.eval()
    emb_layer = _get_embedding_layer(model.semantic_encoder)
    if emb_layer is None:
        raise RuntimeError("Cannot locate token embedding layer for integrated gradients")
    ids       = batch["input_ids"].to(_dev)
    base_emb  = emb_layer(torch.zeros_like(ids)).detach()
    inp_emb   = emb_layer(ids).detach()
    ig        = torch.zeros_like(inp_emb)
    lf        = batch["ling_feats"].to(_dev) if "ling_feats" in batch else None
    for step in range(steps):
        interp = (base_emb + (step / steps) * (inp_emb - base_emb)).requires_grad_(True)
        orig_fwd = emb_layer.forward
        try:
            emb_layer.forward = lambda _: interp
            out = model(input_ids=ids, attention_mask=batch["attention_mask"].to(_dev), ling_feats=lf)
        finally:
            emb_layer.forward = orig_fwd
        out["logits"][:, target_class].sum().backward()
        ig += interp.grad.detach()
    return (ig * (inp_emb - base_emb)).norm(dim=-1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cfg = CFG

    # 1. Data
    train_df, val_df, test_df = load_combined(cfg)
    log.info(f"train={len(train_df)} | val={len(val_df)} | test={len(test_df)} | "
             f"deceptive_train={int(train_df['label'].sum())}")

    # 2. GPT-2 perplexity feature
    if cfg.use_perplexity:
        log.info("Computing GPT-2 perplexity features …")
        all_texts = train_df["text"].tolist() + val_df["text"].tolist() + test_df["text"].tolist()
        all_perp  = _compute_perplexity(all_texts, cfg.device, cfg.perp_model_id, cfg.perp_batch_size)
        n_tr, n_va = len(train_df), len(val_df)
        train_df = train_df.copy(); train_df["perplexity"] = all_perp[:n_tr]
        val_df   = val_df.copy();   val_df["perplexity"]   = all_perp[n_tr : n_tr + n_va]
        test_df  = test_df.copy();  test_df["perplexity"]  = all_perp[n_tr + n_va :]

    # 3. Multi-seed training with snapshot ensemble
    seeds = list(cfg.multi_seed_list) if cfg.multi_seed else [cfg.seed]
    all_snapshots: List[Tuple] = []
    first_tokenizer = None

    for fold_idx, seed in enumerate(seeds):
        cfg.seed = seed; set_seed(seed)
        result = train_fold(train_df, val_df, fold_idx, cfg)
        all_snapshots.extend(result["snapshots"])
        if first_tokenizer is None: first_tokenizer = result["tokenizer"]

    if cfg.multi_seed:
        f1s = [s[0] for s in all_snapshots]
        log.info(f"All snapshots: mean F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    all_snapshots.sort(key=lambda x: x[0], reverse=True)
    top_n        = cfg.snapshot_top_k * len(seeds)
    top_snapshots = all_snapshots[:top_n]
    log.info(f"Ensemble: {len(top_snapshots)} snapshots, best F1={top_snapshots[0][0]:.4f}")

    # 4. Pseudo-labeling round
    if cfg.use_pseudo_label and len(top_snapshots) > 0:
        log.info("Pseudo-labeling …")
        pseudo_model = DeceptionReasoningModel(cfg).to(cfg.device)
        pseudo_model.load_state_dict(top_snapshots[0][2])
        aug_train_df = pseudo_label_round(pseudo_model, test_df, train_df, cfg, first_tokenizer)
        del pseudo_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if len(aug_train_df) > len(train_df):
            pseudo_seeds = list(cfg.multi_seed_list)[:2]
            for fold_idx, seed in enumerate(pseudo_seeds):
                cfg.seed = seed; set_seed(seed)
                result = train_fold(aug_train_df, val_df, len(seeds) + fold_idx, cfg)
                all_snapshots.extend(result["snapshots"])
            all_snapshots.sort(key=lambda x: x[0], reverse=True)
            top_snapshots = all_snapshots[:top_n]
            log.info(f"After pseudo-label retraining: best F1={top_snapshots[0][0]:.4f}")

    # 5. Ensemble inference on val + test
    final_model = DeceptionReasoningModel(cfg).to(cfg.device)
    val_logits  = _ensemble_infer_snapshots(final_model, top_snapshots, val_df,  first_tokenizer, cfg)
    test_logits = _ensemble_infer_snapshots(final_model, top_snapshots, test_df, first_tokenizer, cfg)

    # 6. Temperature calibration on ensemble (fit T on val logits)
    if cfg.use_temperature_scaling:
        val_labels_t = torch.tensor(val_df["label"].tolist())
        T            = nn.Parameter(torch.ones(1))
        opt          = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
        def closure():
            opt.zero_grad()
            F.cross_entropy(val_logits / T.clamp(min=0.1), val_labels_t).backward()
            return F.cross_entropy(val_logits / T.clamp(min=0.1), val_labels_t)
        opt.step(closure)
        T_val       = float(max(0.1, min(10.0, T.item())))
        val_logits  = val_logits  / T_val
        test_logits = test_logits / T_val
        log.info(f"Ensemble temperature: {T_val:.3f}")

    # 7. Save best single model
    best_model = DeceptionReasoningModel(cfg).to(cfg.device)
    best_model.load_state_dict(top_snapshots[0][2])
    model_path = save_model(best_model, cfg, tokenizer=first_tokenizer)

    # 8. Full benchmark evaluation (baselines + ablations + per-domain + TTA)
    log.info("Starting benchmark evaluation …")
    run_evaluation(best_model, first_tokenizer, train_df, val_df, test_df, cfg,
                   skip_semantic_only=True)

    # 9. Threshold on VAL, evaluate on TEST (no leakage)
    val_probs   = torch.softmax(val_logits,  -1)[:, 1].tolist()
    val_labels  = val_df["label"].tolist()
    val_domains = val_df["domain_id"].tolist() if "domain_id" in val_df else [3] * len(val_df)
    thresh = find_optimal_thresholds(val_probs, val_labels, val_domains, cfg.dann_num_domains)

    probs        = torch.softmax(test_logits, -1)[:, 1].tolist()
    preds        = test_logits.argmax(-1).tolist()
    labels       = test_df["label"].tolist()
    domain_ids_t = test_df["domain_id"].tolist() if "domain_id" in test_df else [3] * len(test_df)
    preds_opt    = [1 if p >= thresh.get(d, thresh[-1]) else 0 for p, d in zip(probs, domain_ids_t)]
    log.info(f"\nTest macro-F1 (ensemble, {len(top_snapshots)} models): "
             f"{f1_score(labels, preds_opt, average='macro', zero_division=0):.4f}")
    print(classification_report(labels, preds_opt, target_names=["Truthful","Deceptive"]))
    log.info(f"Done. Model at: {model_path}")


# ── Evaluation pipeline (runs automatically after training) ───────────────────

def _metrics(labels, preds, probs):
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return {
        "acc":    round(accuracy_score(labels, preds), 4),
        "f1":     round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "auc":    round(auc, 4),
        "prec_d": round(precision_score(labels, preds, pos_label=1, zero_division=0), 4),
        "rec_d":  round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
    }


def _eval_infer(model, df, tokenizer, cfg, no_ling=False):
    loader = DataLoader(
        DeceptionDataset(
            df["text"].tolist(), df["label"].tolist(), tokenizer, cfg,
            domain_ids=df["domain_id"].values if "domain_id" in df else None,
        ),
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_g = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            out = model(
                input_ids=batch_g["input_ids"],
                attention_mask=batch_g["attention_mask"],
                ling_feats=None if no_ling else batch_g.get("ling_feats"),
            )
            probs.extend(torch.softmax(out["logits"], -1)[:, 1].cpu().tolist())
            preds.extend(out["logits"].argmax(-1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    return preds, labels, probs


def _eval_run_majority(test_df):
    majority = int(test_df["label"].mode()[0])
    labels   = test_df["label"].tolist()
    return _metrics(labels, [majority] * len(labels), [float(majority)] * len(labels))


def _eval_run_tfidf_lr(train_df, test_df):
    print("  fitting TF-IDF + LR …")
    vec  = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(train_df["text"])
    X_te = vec.transform(test_df["text"])
    clf  = LogisticRegression(max_iter=1_000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, train_df["label"])
    preds = clf.predict(X_te).tolist()
    probs = clf.predict_proba(X_te)[:, 1].tolist()
    return _metrics(test_df["label"].tolist(), preds, probs)


class _SemanticOnlyBaseline(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.drop    = nn.Dropout(0.1)
        self.head    = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def _eval_run_semantic_only(train_df, test_df, cfg, epochs=3):
    print(f"  fine-tuning {cfg.semantic_model_id.split('/')[-1]}-only baseline …")
    tok     = AutoTokenizer.from_pretrained(cfg.semantic_model_id)
    use_amp = cfg.fp16 and cfg.device == "cuda"

    def _enc(df):
        e = tok(df["text"].tolist(), padding=True, truncation=True,
                max_length=cfg.max_seq_len, return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor(df["label"].tolist())

    tr_ids, tr_mask, tr_lbl = _enc(train_df)
    te_ids, te_mask, te_lbl = _enc(test_df)

    tr_loader = DataLoader(TensorDataset(tr_ids, tr_mask, tr_lbl),
                           batch_size=cfg.batch_size * 2, shuffle=True)
    te_loader = DataLoader(TensorDataset(te_ids, te_mask, te_lbl),
                           batch_size=cfg.batch_size * 2, shuffle=False)

    baseline  = _SemanticOnlyBaseline(cfg.semantic_model_id).to(cfg.device)
    baseline.float()
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    w = torch.tensor(
        [len(train_df) / (2 * max(n_neg, 1)), len(train_df) / (2 * max(n_pos, 1))],
        device=cfg.device,
    )
    criterion  = nn.CrossEntropyLoss(weight=w)
    optimizer  = torch.optim.AdamW(baseline.parameters(), lr=2e-5, weight_decay=0.01)
    scaler     = _make_scaler(use_amp)
    total_steps = len(tr_loader) * epochs
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        baseline.train()
        for ids, mask, lbls in tqdm(tr_loader, desc=f"  Semantic-only epoch {epoch+1}/{epochs}", leave=False):
            ids, mask, lbls = ids.to(cfg.device), mask.to(cfg.device), lbls.to(cfg.device)
            with autocast(_AMP_DEVICE, enabled=use_amp):
                loss = criterion(baseline(ids, mask), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    baseline.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for ids, mask, lbls in te_loader:
            ids, mask = ids.to(cfg.device), mask.to(cfg.device)
            with autocast(_AMP_DEVICE, enabled=use_amp):
                logits = baseline(ids, mask)
            probs.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(lbls.tolist())
    return _metrics(labels, preds, probs)


def _eval_run_cdrs(model, test_df, tokenizer, cfg):
    print("  running inference …")
    preds, labels, probs = _eval_infer(model, test_df, tokenizer, cfg)
    return _metrics(labels, preds, probs), preds, labels


def _eval_tune_threshold(val_df, model, tokenizer, cfg):
    print("  tuning threshold on val set …")
    _, labels, probs = _eval_infer(model, val_df, tokenizer, cfg)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.71, 0.01):
        ps = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, ps, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"  optimal threshold: {best_t:.2f}  (val macro-F1: {best_f1:.4f})")
    return best_t


def _eval_apply_threshold(test_df, model, tokenizer, cfg, threshold):
    _, labels, probs = _eval_infer(model, test_df, tokenizer, cfg)
    preds_t = [1 if p >= threshold else 0 for p in probs]
    return _metrics(labels, preds_t, probs)


def _eval_run_ling_ablation(model, test_df, tokenizer, cfg):
    print("  ablation: ling features removed …")
    preds, labels, probs = _eval_infer(model, test_df, tokenizer, cfg, no_ling=True)
    m = _metrics(labels, preds, probs)
    print(f"    no-ling: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return m


def _eval_infer_tta(model, df, tokenizer, cfg):
    variants = [df["text"].tolist(), [t.lower() for t in df["text"]]]
    sum_probs = None
    labels    = None
    for texts in variants:
        aug = df.copy()
        aug["text"] = texts
        _, lbls, probs = _eval_infer(model, aug, tokenizer, cfg)
        arr = np.array(probs)
        sum_probs = arr if sum_probs is None else sum_probs + arr
        labels    = lbls
    avg_probs = (sum_probs / len(variants)).tolist()
    preds     = [1 if p >= 0.5 else 0 for p in avg_probs]
    return preds, labels, avg_probs


def _eval_run_tta(model, test_df, tokenizer, cfg):
    print("  TTA inference (original + lowercase) …")
    preds, labels, probs = _eval_infer_tta(model, test_df, tokenizer, cfg)
    m = _metrics(labels, preds, probs)
    print(f"    TTA: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    return m


def _eval_run_per_domain(model, test_df, tokenizer, cfg):
    print("  evaluating per domain …")
    if "domain" not in test_df.columns:
        print("  (no domain column — skipping)")
        return {}
    results = {}
    for domain in sorted(test_df["domain"].unique()):
        sub   = test_df[test_df["domain"] == domain].reset_index(drop=True)
        preds, labels, probs = _eval_infer(model, sub, tokenizer, cfg)
        m = _metrics(labels, preds, probs)
        results[domain] = m
        print(f"    {domain:25s}  F1={m['f1']:.4f}  Acc={m['acc']:.4f}  n={len(sub)}")
    return results


def _print_eval_table(rows):
    hdr = "{:36s}  {:6s}  {:8s}  {:7s}  {:7s}  {:6s}"
    row = "{:36s}  {:.4f}  {:.4f}    {:.4f}   {:.4f}   {:.4f}"
    print("\n" + "=" * 80)
    print(hdr.format("System", "Acc", "Macro-F1", "AUC-ROC", "Prec(D)", "Rec(D)"))
    print("-" * 80)
    for name, m in rows:
        print(row.format(name, m["acc"], m["f1"], m["auc"], m["prec_d"], m["rec_d"]))
    print("=" * 80)


def run_evaluation(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cfg:      Config,
    skip_semantic_only: bool = False,
) -> dict:
    """Full benchmark evaluation; called automatically at end of training."""
    output_path = os.path.join(cfg.output_dir, "benchmark_results.json")
    results: dict = {}

    print("\n[1] Majority class")
    results["majority"] = _eval_run_majority(test_df)

    print("\n[2] TF-IDF + LR")
    results["tfidf_lr"] = _eval_run_tfidf_lr(train_df, test_df)

    if not skip_semantic_only:
        print(f"\n[3] Semantic-only ({cfg.semantic_model_id.split('/')[-1]} + linear)")
        results["semantic_only"] = _eval_run_semantic_only(train_df, test_df, cfg, epochs=3)
    else:
        results["semantic_only"] = None

    print("\n[4] Full model")
    cdrs_m, cdrs_preds, cdrs_labels = _eval_run_cdrs(model, test_df, tokenizer, cfg)
    results["cdrs"] = cdrs_m

    print("\n[5] Threshold tuning")
    best_t = _eval_tune_threshold(val_df, model, tokenizer, cfg)
    results["cdrs_tuned"]        = _eval_apply_threshold(test_df, model, tokenizer, cfg, best_t)
    results["optimal_threshold"] = best_t

    print("\n[6] Ling-feature ablation")
    results["ablation_no_ling"] = _eval_run_ling_ablation(model, test_df, tokenizer, cfg)

    print("\n[7] Per-domain evaluation")
    results["per_domain"] = _eval_run_per_domain(model, test_df, tokenizer, cfg)

    print("\n[8] Test-time augmentation")
    results["tta"] = _eval_run_tta(model, test_df, tokenizer, cfg)

    table_rows = [
        ("Majority class", results["majority"]),
        ("TF-IDF + LR",    results["tfidf_lr"]),
    ]
    if results["semantic_only"]:
        sem_label = f"Semantic-only ({cfg.semantic_model_id.split('/')[-1]})"
        table_rows.append((sem_label, results["semantic_only"]))
    table_rows.append(("Full model",                   results["cdrs"]))
    table_rows.append((f"Full model (t={best_t:.2f})", results["cdrs_tuned"]))
    table_rows.append(("Full model (no ling)",         results["ablation_no_ling"]))
    table_rows.append(("Full model + TTA",             results["tta"]))
    _print_eval_table(table_rows)

    print("\nDetailed classification report:")
    print(classification_report(cdrs_labels, cdrs_preds, target_names=["Truthful", "Deceptive"]))

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_path}")
    return results


if __name__ == "__main__":
    main()
