"""
bedrock_kb_retrieval.py
=======================
Pre-written helper - Bedrock Knowledge Base retrieval utility.

This module provides a thin wrapper around the Bedrock Agent Runtime
`retrieve()` API with X-Ray remote subsegment tracing so Knowledge Base
nodes appear on the AWS Service Map.
"""

import boto3
import os
import json
from contextlib import contextmanager

# Import X-Ray KB tracer from agent_observability
try:
    from agent_observability import trace_kb_retrieval
except ImportError:
    @contextmanager
    def trace_kb_retrieval(kb_id: str):
        yield None

# Bedrock Agent Runtime client
_bedrock_agent_runtime = boto3.client(
    'bedrock-agent-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)


def retrieve_from_knowledge_base(
    kb_id: str,
    query: str,
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the top-k most relevant document chunks from a Bedrock Knowledge Base.
    """
    if not kb_id:
        return []

    try:
        # Wrap the retrieval in the X-Ray subsegment
        with trace_kb_retrieval(kb_id):
            response = _bedrock_agent_runtime.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": query},
            )
    except Exception as exc:
        # Surface the error as structured text so the calling agent can report it
        return [{
            'text':   f"Knowledge base retrieval failed: {exc}",
            'source': 'error',
            'score':  0.0
        }]

    results = []
    for item in response.get('retrievalResults', []):
        content = item.get('content', {})
        location = item.get('location', {})
        score    = item.get('score', 0.0)

        # Extract text - KB returns either plain text or a structured object
        text = content.get('text', '')

        # Extract S3 URI from location metadata
        s3_location = location.get('s3Location', {})
        source = s3_location.get('uri', 'unknown')

        results.append({
            'text':   text,
            'source': source,
            'score':  round(float(score), 4)
        })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def format_kb_results(results: list[dict]) -> str:
    """
    Format Knowledge Base results into a readable string for an agent's context.
    """
    if not results:
        return "No relevant policy documents found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Passage {i} | Score: {r['score']:.3f}]\n"
            f"{r['text']}\n"
            f"Source: {r['source']}"
        )
    return "\n\n".join(formatted)