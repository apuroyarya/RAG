"""Qdrant access, in embedded or server mode.

Embedded mode (a local directory, no server, no Docker) is the default so the
project keeps running with nothing installed. Set QDRANT_URL to use a server.

One caveat worth knowing before it bites: **embedded mode takes an exclusive
lock on its directory**, so only one process can hold it. The API server and a
CLI script cannot both open it at once - the second gets a "already accessed by
another instance" error. That is fine for phase 1's manual triggers, and it is
the main reason to move to the server for anything concurrent.

The collection uses a *named* dense vector from the start. Qdrant cannot add a
new named vector to an existing collection, and the design wants hybrid search
(dense + sparse) later - so naming it now means adding sparse is a collection
rebuild rather than a schema problem. A rebuild is cheap because the embedding
artifacts are the durable record; nothing has to be re-embedded.
"""
import uuid

from .config import QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL

#: Fixed namespace so a chunk id always maps to the same point id. Qdrant point
#: ids must be uint64 or UUID, and chunk ids are strings like
#: "<doc>:<page>:<ordinal>" - uuid5 makes that mapping deterministic, so
#: re-indexing overwrites the right point instead of adding a duplicate.
POINT_NAMESPACE = uuid.UUID("6f1d3c2e-4a5b-4c7d-8e9f-0a1b2c3d4e5f")

#: Name of the dense vector in the collection. See module docstring.
DENSE = "dense"


def point_id(chunk_id):
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def available():
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        return False, "pip install qdrant-client"
    return True, "ok"


def connect():
    """Returns a QdrantClient. Server if QDRANT_URL is set, else embedded."""
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def collection_info(client, name=None):
    """(exists, dim) for the collection, without raising if it is absent."""
    name = name or QDRANT_COLLECTION
    try:
        info = client.get_collection(name)
    except Exception:
        return False, None
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        params = vectors.get(DENSE)
        return True, (params.size if params else None)
    return True, getattr(vectors, "size", None)


def ensure_collection(client, dim, name=None, recreate=False):
    """Create the collection if needed. Returns True if it was (re)created.

    A dimension mismatch forces a recreate: vectors of the wrong width cannot
    be inserted, and silently keeping the old collection would leave the corpus
    split across two incompatible shapes.
    """
    from qdrant_client.models import Distance, VectorParams

    name = name or QDRANT_COLLECTION
    exists, existing_dim = collection_info(client, name)

    if exists and not recreate and existing_dim == dim:
        return False
    if exists:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        # Cosine, and vectors are already L2-normalized at embed time, so a
        # dot product elsewhere agrees with Qdrant's scoring
        vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
    )
    return True


def upsert_chunks(client, records, name=None):
    """records: iterable of (chunk_id, vector, payload). Deterministic ids."""
    from qdrant_client.models import PointStruct

    name = name or QDRANT_COLLECTION
    points = [
        PointStruct(id=point_id(chunk_id), vector={DENSE: vector}, payload=payload)
        for chunk_id, vector, payload in records
    ]
    if points:
        client.upsert(collection_name=name, points=points, wait=True)
    return len(points)


def delete_document(client, document_id, name=None):
    """Remove one document's points, so re-indexing cannot orphan old chunks.

    Chunk ids are deterministic, so a re-index overwrites points that still
    exist - but if the chunker produced *fewer* chunks this time, the surplus
    points from the previous run would linger and keep being retrieved.
    """
    from qdrant_client.models import (FieldCondition, Filter, FilterSelector,
                                      MatchValue)

    name = name or QDRANT_COLLECTION
    client.delete(
        collection_name=name,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ])),
        wait=True,
    )


def count(client, name=None):
    name = name or QDRANT_COLLECTION
    try:
        return client.count(collection_name=name, exact=True).count
    except Exception:
        return 0


def search(client, vector, limit=20, document_id=None, name=None):
    """Dense search. Returns [(score, payload)] highest first."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    name = name or QDRANT_COLLECTION
    query_filter = None
    if document_id:
        query_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ])
    hits = client.query_points(
        collection_name=name,
        query=vector,
        using=DENSE,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points
    return [(h.score, h.payload) for h in hits]
