from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from collections import Counter
from typing import Any, List, Dict, Optional
import json
import os
from pathlib import Path

class USDAssetStore:
    """USD asset store backed by ChromaDB."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "usd_collection_1218",
        embedding_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the asset store.

        Args:
            persist_directory: ChromaDB persistence directory.
            collection_name: Chroma collection name.
            embedding_config: Embedding configuration dict (model/api_key).
        """
        self.embeddings = self._build_embeddings(embedding_config or {})
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        # LangChain integration method
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def _collection_count(self) -> int:
        """Return current configured collection document count."""
        try:
            col = self.client.get_collection(name=self.collection_name)
            return col.count()
        except Exception:
            return -1

    @staticmethod
    def _resolve_api_key(
        api_key: Optional[str],
        api_key_env: Optional[str] = None,
    ) -> Optional[str]:
        if api_key:
            return api_key
        if api_key_env:
            return os.getenv(api_key_env)
        return None

    @classmethod
    def _build_embeddings(cls, embedding_config: Dict[str, Any]):
        model = embedding_config.get("model")
        api_key = cls._resolve_api_key(
            embedding_config.get("api_key"),
            embedding_config.get("api_key_env"),
        )

        model = model or "text-embedding-v4"
        if not api_key:
            raise ValueError(
                "Qwen embedding requires api_key/api_key_env in run_config.json -> rag.embedding"
            )
        return DashScopeEmbeddings(dashscope_api_key=api_key, model=model)

    @classmethod
    def from_config_file(cls, config_path: str) -> "USDAssetStore":
        """Create store from run_config.json rag section."""
        resolved_config_path = Path(config_path).resolve()
        with resolved_config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        # Reuse centralized env loading so embedding keys can live in one .env file.
        try:
            from llm import load_env_from_run_config

            loaded_env_path = load_env_from_run_config(config, resolved_config_path)
            if loaded_env_path:
                print(f"[RAG] Loaded environment variables from: {loaded_env_path}")
        except Exception as e:
            print(f"Warning: failed to preload env file from config: {e}")

        rag_config = config.get("rag") or {}
        vector_db_config = rag_config.get("vector_db") or {}

        persist_directory_cfg = vector_db_config.get("persist_directory")
        if not persist_directory_cfg:
            raise ValueError("Missing rag.vector_db.persist_directory in run_config.json")
        # Expand ${VAR} / $VAR / ~ so users can configure paths via env vars.
        persist_directory_cfg = os.path.expanduser(
            os.path.expandvars(persist_directory_cfg)
        )

        persist_path = Path(persist_directory_cfg)
        if not persist_path.is_absolute():
            persist_path = (resolved_config_path.parent / persist_path).resolve()

        collection_name = vector_db_config.get("collection_name") or "usd_collection_1218"
        embedding_config = rag_config.get("embedding") or {}

        return cls(
            persist_directory=str(persist_path),
            collection_name=collection_name,
            embedding_config=embedding_config,
        )

    
    def search_with_score(
        self,
        query: str,
        top_k: int = 5,
        exclude_prefixes: List[str] = None,
        exclude_categories: List[str] = None,
        vector_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        dedupe_by_category: bool = True,
        max_per_category: Optional[int] = 1,
    ) -> List[tuple]:
        """
        Search and return similarity scores, excluding duplicate categories.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            exclude_prefixes: List of asset_id prefixes to exclude (default: structural assets).
            exclude_categories: List of categories to exclude (default: structural categories).
            vector_type: Vector type filter.
            fetch_k: Initial fetch count before filtering.
            dedupe_by_category: Whether to deduplicate by category.
            max_per_category: Maximum number of results per category.

        Returns:
            Filtered list of (Document, score) tuples.
        """
        if exclude_prefixes is None:
            exclude_prefixes = ['ceilings', 'floor', 'walls', 'door', 'ceiling', 'wall', 'floors', 'doors']
        if exclude_categories is None:
            exclude_categories = ['door', 'wall', 'floor', 'ceiling', 'walls', 'floors', 'ceilings', 'doors']

        fetch_k = int(fetch_k or max(top_k * 5, top_k))
        search_kwargs: Dict[str, Any] = {"k": fetch_k}
        if vector_type:
            search_kwargs["filter"] = {"vector_type": vector_type}
        results = self.vector_store.similarity_search_with_score(query, **search_kwargs)

        if not results:
            count = self._collection_count()
            print(
                f"[RAG] Empty retrieval for query='{query}'. "
                f"vector_type='{vector_type or 'any'}', collection='{self.collection_name}', "
                f"persist_directory='{self.persist_directory}', count={count}"
            )
            return []

        filtered_results = []
        category_counts = Counter()
        category_limit = 1 if dedupe_by_category else max_per_category
        if category_limit is not None and category_limit <= 0:
            category_limit = None
        exclude_categories_set = set(c.lower() for c in (exclude_categories or []))
        for doc, score in results:
            asset_id = doc.metadata.get('asset_id', '')
            category = doc.metadata.get('category', '')
            if any(asset_id.startswith(prefix) for prefix in exclude_prefixes):
                continue
            if exclude_categories_set and category.lower() in exclude_categories_set:
                continue
            if category_limit is not None and category_counts[category] >= category_limit:
                continue
            filtered_results.append((doc, score))
            category_counts[category] += 1
            if len(filtered_results) >= top_k:
                break

        if not filtered_results:
            # Retrieval had hits but all were filtered out; filters may be too strict.
            print(
                f"[RAG] Retrieval had {len(results)} hits but all were filtered out. "
                f"vector_type='{vector_type or 'any'}', exclude_prefixes={exclude_prefixes}, "
                f"dedupe_by_category={dedupe_by_category}, max_per_category={max_per_category}, "
                f"fetch_k={fetch_k}."
            )

        return filtered_results
    
    # Batch insert: caption is used for embedding, all other fields stored as metadata.
    def insert_assets(self, assets: List[Dict]) -> List[str]:
        """
        Batch insert assets into the Chroma store.
        Each dict in assets must contain at least:
          - asset_id (str)
          - caption (str)  <- used as embedding text
        Optional:
          - usd_path, category, extended_info (dict), etc. — stored as metadata.
        Returns the list of inserted ids.
        """
        if not isinstance(assets, list) or not assets:
            return []

        docs: List[Document] = []
        ids: List[str] = []
        for a in assets:
            aid = str(a.get("asset_id") or a.get("id") or "")
            if not aid:
                raise ValueError("Each asset must contain an asset_id field")
            caption = a.get("caption", "")
            # caption is the text used for embedding
            text = caption
            # All other information is stored as metadata
            metadata = {
                "asset_id": aid,
                "usd_path": a.get("usd_path", ""),
                "category": a.get("category", ""),
                "caption": caption
            }
            docs.append(Document(page_content=text, metadata=metadata))
            ids.append(aid)

        # Prefer the LangChain Chroma wrapper interface for insertion
        try:
            if hasattr(self.vector_store, "add_documents"):
                self.vector_store.add_documents(docs, ids=ids)
            else:
                texts = [d.page_content for d in docs]
                metadatas = [d.metadata for d in docs]
                if hasattr(self.vector_store, "add_texts"):
                    self.vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
                else:
                    raise AttributeError("Chroma wrapper does not support add_documents or add_texts")
        except Exception as e:
            # Fall back to the native chromadb client method
            try:
                col = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.embeddings)
                texts = [d.page_content for d in docs]
                metadatas = [d.metadata for d in docs]
                col.add(ids=ids, documents=texts, metadatas=metadatas)
            except Exception as e2:
                raise RuntimeError(f"Asset insertion failed: {e} ; fallback also failed: {e2}")

        return ids

    def insert_asset(self, asset: Dict) -> str:
        """Insert a single asset and return its asset_id."""
        ids = self.insert_assets([asset])
        return ids[0] if ids else ""

if __name__ == "__main__":
    default_config = Path(__file__).resolve().parents[1] / "run_config.json"
    store = USDAssetStore.from_config_file(str(default_config))
    res = store.search_with_score("cutting board used for food preparation in a kitchen", top_k=10, vector_type="explicit")
    for r in res:
        print(r)

