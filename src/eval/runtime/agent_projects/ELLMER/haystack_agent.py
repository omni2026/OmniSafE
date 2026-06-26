import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers.document_writer import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from prompt import *

script_dir = Path(__file__).parent
default_robot_document_path = script_dir / "custom_knowledge.md"
default_embedding_cache_path = script_dir / "custom_knowledge_embeddings.pkl"

document_store: Optional[InMemoryDocumentStore] = None
rag_pipeline: Optional[Pipeline] = None
messages = []
_agent_runtime: Dict[str, Any] = {}
_completion_capture: Dict[str, Any] = {"response": None}


class _CompletionsCaptureProxy:
    """Record the raw OpenAI-compatible completion before Haystack flattens it."""

    def __init__(self, delegate: Any, capture: Dict[str, Any]):
        self._delegate = delegate
        self._capture = capture

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._delegate.create(*args, **kwargs)
        self._capture["response"] = response
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ChatCaptureProxy:
    def __init__(self, delegate: Any, capture: Dict[str, Any]):
        self._delegate = delegate
        self.completions = _CompletionsCaptureProxy(delegate.completions, capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _OpenAIClientCaptureProxy:
    def __init__(self, delegate: Any, capture: Dict[str, Any]):
        self._delegate = delegate
        self.chat = _ChatCaptureProxy(delegate.chat, capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def resolve_llm_config(llm_config: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not llm_config:
        raise ValueError(
            f"{label} is required. Provider, model, api_key, and base_url "
            "must be configured on the Eval side and passed into ELLMER."
        )

    resolved = dict(llm_config)
    if not resolved.get("model"):
        raise ValueError(f"{label}.model is required for ELLMER.")
    if not resolved.get("api_key"):
        raise ValueError(f"{label}.api_key is required for ELLMER.")
    if not resolved.get("base_url"):
        raise ValueError(f"{label}.base_url is required for ELLMER.")
    return resolved


def _set_openai_env(api_key: str, base_url: str) -> None:
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url


def compute_file_hash(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def load_source_documents(file_path: Path) -> list[Document]:
    content = file_path.read_text(encoding="utf-8")
    return [Document(content=content)]


def document_to_cache_dict(doc: Document) -> dict:
    return doc.to_dict()


def load_cached_documents(cache_path: Path, expected_source_hash: str, expected_model: str):
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None

    if not isinstance(payload, dict):
        return None

    if payload.get("source_hash") != expected_source_hash:
        return None

    if payload.get("embedding_model") != expected_model:
        return None

    documents_data = payload.get("documents", [])
    if not documents_data:
        return None

    return [Document.from_dict(item) for item in documents_data]


def save_cached_documents(cache_path: Path, source_file: Path, source_hash: str, embedding_model: str, documents: list[Document]) -> None:
    payload = {
        "source_file": str(source_file),
        "source_hash": source_hash,
        "embedding_model": embedding_model,
        "documents": [document_to_cache_dict(doc) for doc in documents],
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_prompt_template() -> str:
    return """
    You are a helpful assistant who interacts naturally with the user while also generating executable Python code for a robot backend. Your responses must contain two distinct parts:

    1. **User-facing response**: A natural, human-like reply to the user's request. Explain what you're doing or what the outcome will be, but never mention code, code blocks, or implementation details. Act as if you're another human carrying out the task directly.

    2. **Execution code**: A separate Python code block that implements the necessary actions for the robot. This code must be enclosed in a markdown Python code block (i.e., ```python ... ```) and placed at the end of your response. This code is NOT shown to the user, it will be parsed and executed by the system.

    Important rules:
    - Never refer to the code block in your user-facing message.
    - Never say things like "I'll run this code" or "Here is the code." Just act.
    - The action (via code) should logically match your explanation.
    - Always place the ```python code block at the very end, and only include executable code, no comments or explanations inside the code block unless required for functionality.

    Now, Given these documents, answer the question.

    Context:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}

    {% if environment_context %}
    Environment:
    {{ environment_context }}
    {% endif %}

    Question: {{query}}
    Answer:
    """


def initialize_agent(
    llm_config: Optional[Dict[str, Any]] = None,
    embedding_llm_config: Optional[Dict[str, Any]] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global document_store, rag_pipeline, messages, _agent_runtime, _completion_capture

    resolved_llm_config = resolve_llm_config(llm_config, "llm_config")
    resolved_embedding_config = resolve_llm_config(
        embedding_llm_config or llm_config,
        "embedding_llm_config",
    )

    runtime = dict(runtime_overrides or {})
    source_path = Path(runtime.get("robot_document_path", default_robot_document_path))
    cache_path = Path(runtime.get("embedding_cache_path", default_embedding_cache_path))
    prompt_template = str(runtime.get("prompt_template") or build_prompt_template())
    llm_timeout_sec = float(runtime.get("llm_timeout_sec", os.getenv("OPENAI_TIMEOUT", "120")))
    llm_max_retries = int(runtime.get("llm_max_retries", os.getenv("OPENAI_MAX_RETRIES", "5")))

    document_store = InMemoryDocumentStore()
    source_hash = compute_file_hash(source_path)

    _set_openai_env(
        resolved_embedding_config["api_key"],
        resolved_embedding_config["base_url"],
    )

    cached_documents = load_cached_documents(
        cache_path,
        expected_source_hash=source_hash,
        expected_model=resolved_embedding_config["model"],
    )

    if cached_documents is not None:
        document_store.write_documents(cached_documents)
        print(f"Loaded cached embeddings from: {cache_path}")
        print(
            "[DEBUG][ELLMER][CACHE_HIT] "
            f"embedding_model={resolved_embedding_config['model']} "
            f"documents={len(cached_documents)} "
            f"source_hash={source_hash} "
            f"cache_path={cache_path}"
        )
    else:
        documents = load_source_documents(source_path)

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "embedder",
            OpenAIDocumentEmbedder(
                model=resolved_embedding_config["model"],
                api_base_url=resolved_embedding_config["base_url"],
            ),
        )
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        indexing_pipeline.connect("embedder.documents", "writer.documents")
        indexing_pipeline.run({"embedder": {"documents": documents}})

        embedded_documents = document_store.filter_documents()
        save_cached_documents(
            cache_path,
            source_file=source_path,
            source_hash=source_hash,
            embedding_model=resolved_embedding_config["model"],
            documents=embedded_documents,
        )
        print(f"Created and saved embeddings to: {cache_path}")

    rag_pipeline = Pipeline()

    _set_openai_env(
        resolved_embedding_config["api_key"],
        resolved_embedding_config["base_url"],
    )
    rag_pipeline.add_component(
        "text_embedder",
        OpenAITextEmbedder(
            model=resolved_embedding_config["model"],
            api_base_url=resolved_embedding_config["base_url"],
        ),
    )
    rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    rag_pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template=prompt_template,
            required_variables=["documents", "query"],
        ),
    )

    _set_openai_env(
        resolved_llm_config["api_key"],
        resolved_llm_config["base_url"],
    )
    llm_generator = OpenAIGenerator(
        model=resolved_llm_config["model"],
        api_base_url=resolved_llm_config["base_url"],
        timeout=llm_timeout_sec,
        max_retries=llm_max_retries,
    )
    _completion_capture = {"response": None}
    llm_generator.client = _OpenAIClientCaptureProxy(
        llm_generator.client,
        _completion_capture,
    )
    rag_pipeline.add_component("llm", llm_generator)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    messages = [
        ChatMessage.from_system(
            "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        )
    ]

    _agent_runtime = {
        "llm_config": resolved_llm_config,
        "embedding_llm_config": resolved_embedding_config,
        "runtime_overrides": runtime,
        "llm_timeout_sec": llm_timeout_sec,
        "llm_max_retries": llm_max_retries,
        "robot_document_path": str(source_path),
        "embedding_cache_path": str(cache_path),
    }
    return dict(_agent_runtime)


def initialize_agent_from_env() -> bool:
    llm_model = os.getenv("ELLMER_LLM_MODEL") or os.getenv("OPENAI_MODEL")
    llm_api_key = os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_API_BASE")

    embedding_model = os.getenv("ELLMER_EMBEDDING_MODEL") or llm_model
    embedding_api_key = os.getenv("ELLMER_EMBEDDING_API_KEY") or llm_api_key
    embedding_base_url = os.getenv("ELLMER_EMBEDDING_BASE_URL") or llm_base_url

    if not all([llm_model, llm_api_key, llm_base_url, embedding_model, embedding_api_key, embedding_base_url]):
        return False

    initialize_agent(
        llm_config={
            "provider": os.getenv("ELLMER_LLM_PROVIDER", "env"),
            "model": llm_model,
            "api_key": llm_api_key,
            "base_url": llm_base_url,
        },
        embedding_llm_config={
            "provider": os.getenv("ELLMER_EMBEDDING_PROVIDER", "env"),
            "model": embedding_model,
            "api_key": embedding_api_key,
            "base_url": embedding_base_url,
        },
    )
    return True


def rag_pipeline_func(query: str, environment_context: str = ""):
    if rag_pipeline is None:
        raise RuntimeError("ELLMER agent is not initialized. Call initialize_agent() first.")

    prompt_builder_data: Dict[str, Any] = {"query": query}
    if environment_context:
        prompt_builder_data["environment_context"] = environment_context

    _completion_capture["response"] = None
    result = rag_pipeline.run(
        data={"prompt_builder": prompt_builder_data, "text_embedder": {"text": query}}
    )
    return {"reply": result["llm"]["replies"][0]}


def get_last_llm_response() -> Any:
    """Return the unmodified completion used for the latest RAG answer."""
    return _completion_capture.get("response")


def chatbot_with_fc(message, history=None, environment_context: str = ""):
    if rag_pipeline is None:
        raise RuntimeError("ELLMER agent is not initialized. Call initialize_agent() first.")

    messages.append(ChatMessage.from_user(message))
    response = rag_pipeline_func(message, environment_context=environment_context)

    messages.append(ChatMessage.from_assistant(response["reply"]))

    return response["reply"]


def main():
    if rag_pipeline is None and not initialize_agent_from_env():
        print("ELLMER agent is not initialized. Use the Eval adapter or set environment variables before running this script directly.")
        return

    print("Kinova CLI - type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting.")
                break
            reply = chatbot_with_fc(user_input)
            print(f"Assistant: {reply}\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
