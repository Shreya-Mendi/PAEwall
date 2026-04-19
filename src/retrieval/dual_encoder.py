"""
Dual-encoder retrieval model for patent-to-product matching.

Wraps the trained DualEncoder model with a FAISS index for fast inference.
Falls back to classical (TF-IDF+LR) then BM25 if the dual encoder is not yet trained.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class RetrievalEngine:
    """
    Production retrieval engine.

    Load priority: dual_encoder+FAISS -> classical TF-IDF+LR -> BM25.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._retriever = None
        self._retriever_type: str | None = None
        self._product_corpus: pd.DataFrame | None = None
        # Dual-encoder specific
        self._faiss_index = None
        self._product_embeddings_ids: list[str] = []
        self._product_embeddings_names: list[str] = []
        self._product_descriptions: list[str] = []
        self._patent_model = None

    def load(self) -> str:
        """
        Load the best available retrieval model.
        Returns the model type that was loaded.
        """
        faiss_path = self.model_dir / "dual_encoder" / "product_index.faiss"
        model_path = self.model_dir / "dual_encoder" / "best_model.pt"
        proj_path  = self.model_dir / "dual_encoder" / "proj_only.pt"
        # proj_only.pt requires re-downloading both 440 MB backbones at startup;
        # skip it here and let classical handle retrieval instead.
        # best_model.pt embeds both towers already — use that when available.
        if not model_path.exists() and proj_path.exists():
            model_path = proj_path   # kept for completeness; see load below
        classical_path = self.model_dir / "classical" / "tfidf_logreg.pkl"
        bm25_path = self.model_dir / "naive" / "bm25.pkl"

        # Training scripts run as __main__, so pickle serialises class paths as
        # __main__.ClassicalRetriever / __main__.BM25Retriever.  When loaded from
        # uvicorn, __main__ is uvicorn itself.  Use a custom Unpickler that
        # redirects __main__ lookups to the correct script module.
        import sys
        import importlib
        scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        class _ScriptUnpickler(pickle.Unpickler):
            _CLASS_MAP = {
                "ClassicalRetriever": "train_classical",
                "BM25Retriever":      "train_naive",
            }
            def find_class(self, module, name):
                if module == "__main__" and name in self._CLASS_MAP:
                    mod = importlib.import_module(self._CLASS_MAP[name])
                    return getattr(mod, name)
                return super().find_class(module, name)

        def _load_pkl(path):
            with open(path, "rb") as f:
                return _ScriptUnpickler(f).load()

        # Only use dual encoder if best_model.pt (full weights) is present.
        # proj_only.pt requires downloading both 440 MB backbones at runtime
        # and is only used when best_model.pt also exists.
        full_model_ready = (self.model_dir / "dual_encoder" / "best_model.pt").exists()
        if faiss_path.exists() and full_model_ready:
            self._load_dual_encoder(model_path, faiss_path)
            self._retriever_type = "dual_encoder"
        elif classical_path.exists():
            self._retriever = _load_pkl(classical_path)
            self._retriever_type = "classical"
        elif bm25_path.exists():
            self._retriever = _load_pkl(bm25_path)
            self._retriever_type = "bm25"
        else:
            raise FileNotFoundError(
                "No trained retrieval model found in models/. "
                "Run train_naive.py, train_classical.py, or the dual-encoder notebook first."
            )

        return self._retriever_type

    def load_product_corpus(self, corpus: pd.DataFrame):
        """Provide the product corpus for classical/BM25 models."""
        self._product_corpus = corpus

    def _load_dual_encoder(self, model_path: Path, faiss_path: Path):
        """Load the dual encoder checkpoint and FAISS index."""
        import faiss
        import torch

        self._faiss_index = faiss.read_index(str(faiss_path))

        # Load product ID mapping alongside the index
        meta_path = faiss_path.parent / "product_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._product_embeddings_ids = meta.get("ids", [])
            self._product_embeddings_names = meta.get("names", [])
            self._product_descriptions = meta.get("descriptions", [])

        # Load encoder for encode_patent().
        # Supports two checkpoint formats:
        #   - full best_model.pt  (model_state key, both backbones included)
        #   - proj_only.pt        (patent_proj + product_proj + log_temperature only;
        #                          backbones downloaded fresh from HuggingFace)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

        # Support two checkpoint formats detected by key presence:
        #   full best_model.pt  → has "model_state" key
        #   proj_only.pt        → has "patent_proj" key (projection heads only)
        proj_path = model_path.parent / "proj_only.pt"
        candidates = [model_path]
        if proj_path.exists() and proj_path != model_path:
            candidates.append(proj_path)

        try:
            from train_deep_learning import DualEncoder
            checkpoint = None
            for cand in candidates:
                if cand.exists():
                    checkpoint = torch.load(cand, map_location="cpu", weights_only=False)
                    break
            if checkpoint is None:
                raise FileNotFoundError("No valid checkpoint found")

            model = DualEncoder()
            if "patent_proj" in checkpoint:
                # proj_only.pt saved from notebook DualEncoder (patent_proj / product_proj attrs).
                # Script DualEncoder uses patent_tower.projection / product_tower.projection.
                # Load projection heads by direct attribute — works for both layouts.
                proj_target = getattr(model, "patent_proj",
                              getattr(model, "patent_tower", None))
                if hasattr(proj_target, "projection"):
                    proj_target.projection.load_state_dict(checkpoint["patent_proj"])
                    model.product_tower.projection.load_state_dict(checkpoint["product_proj"])
                else:
                    model.patent_proj.load_state_dict(checkpoint["patent_proj"])
                    model.product_proj.load_state_dict(checkpoint["product_proj"])
                if "log_temperature" in checkpoint:
                    with torch.no_grad():
                        model.log_temperature.copy_(checkpoint["log_temperature"])
            else:
                state = checkpoint.get("model_state", checkpoint)
                model.load_state_dict(state)
            model.eval()
            self._patent_model = model
        except Exception:
            self._patent_model = None

    def encode_patent(self, claim_text: str, figure_path: str | None = None) -> np.ndarray:
        """Encode a patent into the shared embedding space (dual encoder only)."""
        if self._retriever_type != "dual_encoder" or self._patent_model is None:
            raise RuntimeError("encode_patent requires a loaded dual encoder.")

        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa")
        enc = tokenizer(claim_text[:1024], max_length=256, truncation=True,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            emb = self._patent_model.encode_patents(enc["input_ids"], enc["attention_mask"])
        arr = emb.cpu().numpy().astype("float32")
        arr = arr / (np.linalg.norm(arr) + 1e-10)
        return arr

    def retrieve(self, claim_text: str, top_k: int = 50) -> list[dict]:
        """
        Retrieve top-K candidate products for a patent query.

        Returns list of dicts: rank, product_id, company_name, score, description.
        """
        if self._retriever_type == "dual_encoder":
            return self._retrieve_dual(claim_text, top_k)
        elif self._retriever_type == "classical":
            if self._product_corpus is None:
                raise RuntimeError("Call load_product_corpus() before retrieve().")
            raw = self._retriever.rank_products(claim_text, self._product_corpus, top_k)
            return self._enrich(raw)
        elif self._retriever_type == "bm25":
            raw = self._retriever.predict(claim_text, top_k)
            return raw
        else:
            raise RuntimeError("No model loaded. Call load() first.")

    def _retrieve_dual(self, claim_text: str, top_k: int) -> list[dict]:
        query = self.encode_patent(claim_text).reshape(1, -1)
        scores, indices = self._faiss_index.search(query, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            results.append({
                "rank": rank + 1,
                "product_id": self._product_embeddings_ids[idx] if idx < len(self._product_embeddings_ids) else str(idx),
                "company_name": self._product_embeddings_names[idx] if idx < len(self._product_embeddings_names) else f"company_{idx}",
                "score": float(score),
                "description": self._product_descriptions[idx] if idx < len(self._product_descriptions) else "",
            })
        return results

    def _enrich(self, raw: list[dict]) -> list[dict]:
        """Add description field from corpus if not already present."""
        if self._product_corpus is None:
            return raw
        desc_map = dict(zip(
            self._product_corpus["company_name"].str.lower(),
            self._product_corpus["product_description"].fillna(""),
        ))
        for r in raw:
            r.setdefault("description", desc_map.get(r["company_name"].lower(), ""))
        return raw

    def build_index(self, product_corpus: list[dict]):
        """
        Build FAISS index over product corpus embeddings.
        Used after dual-encoder training to pre-compute the product index.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        descriptions = [p.get("business_description", "") for p in product_corpus]
        embeddings = encoder.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        out_dir = self.model_dir / "dual_encoder"
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / "product_index.faiss"))

        meta = {
            "ids": [str(i) for i in range(len(product_corpus))],
            "names": [p.get("company_name", "") for p in product_corpus],
            "descriptions": descriptions,
        }
        with open(out_dir / "product_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
