---
layout: post
title: "Do You Actually Need LangChain to Build RAG?"
subtitle: "Notes from my RAG learning path on frameworks, plumbing, and knowing what you're trading away."
date: 2026-07-15 09:00:00 -0700
categories: [AI Systems]
tags: [AI Systems, RAG, LangChain]
permalink: /technical-blog/rag-langchain-vs-diy.html
summary: "A practical breakdown of when LangChain helps, when custom RAG code is better, and how to choose between speed, control, and long-term maintainability."
---

While working through my RAG learning path, one question kept resurfacing: **should I build with LangChain, or just write the pipeline myself?**

It sounds like a tooling debate, but it's really a question about trade-offs — speed vs. control, convenience vs. transparency. In this post I'll break down how RAG actually works under the hood, what LangChain gives you, why many production teams skip it, and a practical framework for deciding which path fits your situation.

## First, a quick framing: what RAG actually is

RAG (Retrieval-Augmented Generation) has exactly two moving parts:

1. **Retrieval** — find the chunks of *your own data* that are relevant to a user's question.
2. **Generation** — hand those chunks to an LLM as context so it can generate a grounded answer.

Everything else — document loaders, text splitters, embedding models, vector databases, prompt templates — is plumbing that serves those two steps.

That plumbing is where the framework question lives. **LangChain** is a toolkit that gives you pre-built pieces for every step: loaders for dozens of file formats, splitters with configurable chunking strategies, connectors for nearly every vector database, prompt templates, chains, and agents. Alternatively, you can write that plumbing yourself with plain Python, an LLM API, and a vector database SDK.

Both approaches get you a working RAG system. They just trade off very differently.

## The pipeline is simpler than it looks

Here's the thing that surprised me most when I dug in: a basic RAG pipeline is genuinely just five steps.

```
embed text -> store vectors -> search -> stuff into a prompt -> call the LLM
```

In plain Python, that's maybe 50 lines. A minimal sketch:

```python
# 1. Chunk your documents
chunks = [doc[i:i+1000] for i in range(0, len(doc), 800)]  # overlap of 200

# 2. Embed and store
embeddings = embed_model.embed(chunks)
vector_db.upsert(zip(chunk_ids, embeddings, chunks))

# 3. At query time: embed the question, search
query_vec = embed_model.embed([question])[0]
top_chunks = vector_db.search(query_vec, top_k=5)

# 4. Stuff into a prompt
prompt = f"""Answer using only the context below.

Context:
{chr(10).join(c.text for c in top_chunks)}

Question: {question}"""

# 5. Call the LLM
answer = llm.generate(prompt)
```

That's the whole architecture. Every RAG system — from a weekend prototype to an enterprise deployment — is an elaboration of these five steps. Once you see this, the framework question becomes much clearer: **what is LangChain adding on top of this, and is that addition worth its cost for *your* use case?**

## Why many teams skip LangChain

### 1. "Magic" black boxes

LangChain wraps each step in classes and chains that are convenient — but they hide what's actually being sent to the LLM. The exact prompt. The exact retrieved text. The exact order of operations.

When your RAG system gives a weird answer (and it will), the first debugging question is always: *what did the LLM actually see?* With hand-rolled code, that's a print statement away. With a framework, it can mean digging through layers of abstraction to reconstruct the final prompt. "Why did the LLM get this weird prompt?" is a much harder question to answer when you didn't build the prompt.

### 2. A heavy dependency for a simple job

If your pipeline really is the five steps above, pulling in a large framework is overkill. LangChain brings its own conceptual vocabulary (chains, runnables, retrievers, agents) that you have to learn *on top of* RAG concepts — and its API has historically changed significantly release to release. That versioning churn is a real maintenance cost: code that worked six months ago may need rewriting after an upgrade.

### 3. Performance and control

Production teams optimizing for cost, latency, and accuracy want fine-grained control over:

- **Chunking strategy** — chunk size and overlap dramatically affect retrieval quality
- **Retrieval scoring** — hybrid search, re-ranking, metadata filtering
- **Caching** — of embeddings, of retrieval results, of LLM responses
- **Prompt formatting** — exactly how context is presented to the model

You *can* customize these in LangChain, but you're often fighting the framework's opinions about how each step should work. Rolling your own means every one of these knobs is directly in your hands.

### 4. Fewer moving parts to break

Fewer dependencies means fewer breaking changes when a library updates, a smaller install footprint, and a codebase that's easier to reason about in production. For teams that carry the pager for their RAG system, that simplicity has real operational value.

## What LangChain genuinely gets right

To be fair to the framework — because this isn't a "LangChain bad" post — there are situations where it clearly earns its place.

**Speed to build.** Prebuilt connectors for dozens of vector databases, document loaders for every file format you'll encounter, and ready-made prompt templates mean you can have a working prototype in an afternoon. Writing that glue code yourself is slower, and much of it is undifferentiated plumbing.

**Provider flexibility.** If you need to swap between vector databases or LLM providers — say, benchmarking Pinecone vs. pgvector, or OpenAI vs. Anthropic models — LangChain's abstractions make that a configuration change instead of a rewrite.

**Advanced patterns without reinventing them.** Agents, multi-step reasoning, conversation memory, and tracing (via LangSmith) come built in. If your roadmap includes these, the framework gives you a running start.

**A guided structure for learning.** For beginners, LangChain's opinionated structure can actually be pedagogically useful — it shows you the standard shape of a RAG pipeline before you've developed opinions of your own.

## Side by side

| Dimension | With LangChain | Without (custom) |
|---|---|---|
| **Speed to build** | Fast — prebuilt connectors, loaders, templates | Slower — you write the glue code |
| **Learning curve** | Steeper — framework concepts on top of RAG concepts | Easier — every step is explicit |
| **Flexibility** | Good for standard patterns; deep customization is harder | Full control over every step |
| **Debuggability** | Errors can be buried in framework internals | Trace exactly what's happening |
| **Maintenance** | Tied to LangChain's release cycle and breaking changes | You control your code's lifecycle |
| **Ecosystem** | Plug-and-play with many tools (agents, memory, LangSmith) | You build or integrate what you need |

## Limitations — on both sides

**LangChain's limitations:** runtime overhead from abstraction layers; a history of significant breaking API changes between versions; a tendency to make simple things feel complicated; and difficult debugging when the prompt the LLM received isn't the prompt you expected.

**Custom code's limitations:** more code to write and maintain yourself; you must handle the edge cases a framework would've handled for you (retries, rate limits, exotic file formats, chunking edge cases); and it takes more upfront engineering time before you have anything working.

Neither list is disqualifying. They're just different bills, paid at different times: the framework bill comes due later (in debugging and upgrades), while the custom bill comes due upfront (in engineering hours).

## A decision framework

Instead of "which is better," ask where you are in the build-vs-optimize journey:

**Reach for LangChain when you are:**
- Prototyping quickly and iteration speed matters more than precision
- Building something that needs to swap between vector databases or LLM providers
- Planning to use advanced patterns (agents, memory, multi-step reasoning) that the framework provides
- Learning RAG and want a guided structure to explore standard patterns

**Roll your own when you are:**
- Optimizing a production system where every millisecond and dollar counts
- Running a pipeline simple enough that a framework adds more overhead than value
- Prioritizing full ownership and understanding of the pipeline for reliability
- Committed to long-term maintenance and want to control your own upgrade path

There's also a hybrid path worth mentioning: many teams prototype with LangChain to validate the use case, then rewrite the production pipeline in plain code once they know exactly what they need. The prototype teaches you which knobs matter; the rewrite gives you clean ownership of them.

## Bottom line

LangChain trades some control and debuggability for speed and built-in features. Rolling your own trades development time for transparency, control, and a leaner system.

Neither is objectively better. The right answer depends on whether you're proving an idea or hardening a system — and being honest with yourself about which one you're actually doing.

---

*This post is part of my ongoing RAG learning path, where I document what I learn as I go deeper into retrieval-augmented generation and LLM system design. If you've faced this decision on your own team, I'd love to hear which way you went — and whether you'd choose the same again.*
