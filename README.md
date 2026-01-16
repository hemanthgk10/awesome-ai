# Awesome AI Engineering Tools [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of tools, frameworks, and platforms for building AI products and agents in production.

## What is this?

Building AI applications in production requires navigating a complex ecosystem of tools across multiple layers - from LLM providers and agent frameworks to vector databases, observability platforms, and deployment infrastructure. This repository serves as a comprehensive reference for engineers who are building AI-powered products.

Whether you're building a simple RAG application or a complex multi-agent system, this list helps you discover and evaluate tools across the entire AI engineering stack:

- **Choosing the right tools** - Compare options for each layer of your AI stack
- **Understanding the landscape** - See how different tools fit together in production architectures
- **Staying current** - Track the rapidly evolving AI tooling ecosystem

## Contents

- [Agent Frameworks](#agent-frameworks)
- [LLM Providers](#llm-providers)
- [Vector Databases](#vector-databases)
- [Memory Systems](#memory-systems)
- [RAG Tools](#rag-tools)
- [MCP Servers & Tools](#mcp-servers--tools)
- [Observability & Monitoring](#observability--monitoring)
- [Evaluation & Testing](#evaluation--testing)
- [Guardrails & Safety](#guardrails--safety)
- [Infrastructure & Compute](#infrastructure--compute)
- [Deployment Platforms](#deployment-platforms)
- [Data Processing & ETL](#data-processing--etl)
- [Feature Stores](#feature-stores)
- [Prompt Management](#prompt-management)
- [Fine-tuning Platforms](#fine-tuning-platforms)
- [Embedding Models](#embedding-models)
- [Workflow Orchestration](#workflow-orchestration)
- [Local LLM Tools](#local-llm-tools)
- [Code Context Tools](#code-context-tools)
- [AI Coding Assistants](#ai-coding-assistants)
- [Vibe Coding Tools](#vibe-coding-tools)
- [AI Specification Tools](#ai-specification-tools)

---

## Agent Frameworks

Frameworks for building autonomous AI agents with control flows, tool use, and multi-agent orchestration.

| Tool | Description | Links |
|------|-------------|-------|
| **CrewAI** | Python framework for orchestrating role-playing, autonomous AI agents that collaborate through crews and flows | [Website](https://www.crewai.com) · [GitHub](https://github.com/crewAIInc/crewAI) |
| **LangChain** | Open-source framework for building AI agents with high-level abstractions and 1000+ integrations | [Website](https://www.langchain.com) · [GitHub](https://github.com/langchain-ai/langchain) |
| **LangGraph** | Low-level orchestration framework for building stateful agents with persistence and human oversight | [Website](https://langchain-ai.github.io/langgraph/) · [GitHub](https://github.com/langchain-ai/langgraph) |
| **AutoGen** | Microsoft's framework for multi-agent AI applications with AgentChat and no-code AutoGen Studio | [Website](https://microsoft.github.io/autogen/) · [GitHub](https://github.com/microsoft/autogen) |
| **Haystack** | Open-source AI framework for production-ready RAG and agents with modular architecture | [Website](https://haystack.deepset.ai) · [GitHub](https://github.com/deepset-ai/haystack) |
| **Pydantic AI** | Python agent framework with FastAPI-like developer experience, type-safety and structured outputs | [GitHub](https://github.com/pydantic/pydantic-ai) |
| **Smolagents** | Hugging Face's library for agents that execute actions as code with minimal overhead | [GitHub](https://github.com/huggingface/smolagents) |
| **Agno** | Framework and runtime for multi-agent systems with memory, knowledge, and guardrails (formerly Phidata) | [GitHub](https://github.com/agno-agi/agno) |
| **Dify** | Agentic workflow builder for constructing and managing agent-based workflows | [Website](https://dify.ai) · [GitHub](https://github.com/langgenius/dify) |
| **AgentGPT** | Browser-based no-code platform for building autonomous agents | [Website](https://agentgpt.reworkd.ai) · [GitHub](https://github.com/reworkd/AgentGPT) |
| **AgentForge** | Low-code platform for creating, testing, and iterating AI-powered autonomous agents | [GitHub](https://github.com/DataBassGit/AgentForge) |
| **OpenAI Agents SDK** | Official SDK for building agents with OpenAI models | [Docs](https://platform.openai.com/docs/guides/agents) |

## LLM Providers

Core AI model providers that power reasoning, generation, and understanding.

| Provider | Description | Links |
|----------|-------------|-------|
| **OpenAI** | Creator of GPT models, ChatGPT, and o1/o3 reasoning models | [Website](https://openai.com) · [API](https://platform.openai.com) |
| **Anthropic** | AI safety company offering Claude models (Opus 4.5, Sonnet, Haiku) | [Website](https://www.anthropic.com) · [API](https://console.anthropic.com) |
| **Google AI** | Gemini model family including Gemini 2.0 and open-source Gemma models | [Website](https://ai.google.dev) · [AI Studio](https://aistudio.google.com) |
| **Mistral AI** | European AI company with privacy-first approach and open-source models | [Website](https://mistral.ai) · [API](https://console.mistral.ai) |
| **Cohere** | Enterprise AI with Command, Embed, and Rerank models for RAG | [Website](https://cohere.com) · [API](https://dashboard.cohere.com) |
| **Meta AI** | Llama open-source model family | [Website](https://ai.meta.com) · [Llama](https://llama.meta.com) |
| **Hugging Face** | Community platform with 2M+ models and inference endpoints | [Website](https://huggingface.co) |
| **AWS Bedrock** | Multi-model access to Claude, Llama, Titan and more | [Website](https://aws.amazon.com/bedrock/) |
| **Azure OpenAI** | Enterprise OpenAI models on Azure infrastructure | [Website](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |
| **Groq** | Ultra-fast inference with custom LPU hardware | [Website](https://groq.com) |
| **Fireworks AI** | Fast inference platform for open-source and custom models | [Website](https://fireworks.ai) |

## Vector Databases

Databases for storing embeddings and enabling semantic search.

| Tool | Description | Links |
|------|-------------|-------|
| **Pinecone** | Managed, serverless vector database for semantic search and RAG at scale | [Website](https://www.pinecone.io) |
| **Weaviate** | AI-native vector database with hybrid search and built-in embedding services | [Website](https://weaviate.io) · [GitHub](https://github.com/weaviate/weaviate) |
| **Qdrant** | Open-source vector database in Rust for fast, scalable similarity search | [Website](https://qdrant.tech) · [GitHub](https://github.com/qdrant/qdrant) |
| **Milvus** | Open-source vector database for scalable similarity search | [Website](https://milvus.io) · [GitHub](https://github.com/milvus-io/milvus) |
| **Chroma** | Open-source embedding database with vector, full-text, and metadata search | [Website](https://www.trychroma.com) · [GitHub](https://github.com/chroma-core/chroma) |
| **pgvector** | Open-source vector similarity search extension for PostgreSQL | [GitHub](https://github.com/pgvector/pgvector) |
| **LanceDB** | Serverless vector database for AI applications with multimodal support | [Website](https://lancedb.com) · [GitHub](https://github.com/lancedb/lancedb) |
| **Vespa** | Big data serving engine with vector search, structured data, and ML ranking | [Website](https://vespa.ai) · [GitHub](https://github.com/vespa-engine/vespa) |
| **OpenSearch** | Open-source search and analytics suite with k-NN vector search powered by FAISS and Lucene | [Website](https://opensearch.org) · [GitHub](https://github.com/opensearch-project/OpenSearch) |

## Memory Systems

Infrastructure for maintaining context, conversation history, and learned information.

| Tool | Description | Links |
|------|-------------|-------|
| **Zep** | Context engineering platform using temporal knowledge graphs for chat history and user behavior | [Website](https://www.getzep.com) · [GitHub](https://github.com/getzep/zep) |
| **Mem0** | Memory layer for AI applications enabling persistent memory across interactions | [Website](https://mem0.ai) · [GitHub](https://github.com/mem0ai/mem0) |
| **Supermemory** | Universal Memory API for AI apps with unified storage and access | [Website](https://supermemory.ai) · [GitHub](https://github.com/supermemory/supermemory) |
| **MemGPT (Letta)** | System enabling LLMs to manage their own memory for extended context | [Website](https://letta.com) · [GitHub](https://github.com/letta-ai/letta) |

## RAG Tools

Frameworks and tools for Retrieval-Augmented Generation.

| Tool | Description | Links |
|------|-------------|-------|
| **LlamaIndex** | Enterprise platform for agentic RAG, document processing, and workflow orchestration | [Website](https://www.llamaindex.ai) · [GitHub](https://github.com/run-llama/llama_index) |
| **LangChain** | RAG primitives with document loaders, text splitters, and retrievers | [Website](https://www.langchain.com) · [GitHub](https://github.com/langchain-ai/langchain) |
| **Haystack** | Modular framework for building production RAG pipelines | [Website](https://haystack.deepset.ai) · [GitHub](https://github.com/deepset-ai/haystack) |
| **Unstructured** | Data extraction from 64+ file types into AI-ready formats | [Website](https://unstructured.io) · [GitHub](https://github.com/Unstructured-IO/unstructured) |
| **Firecrawl** | Web scraping and crawling API optimized for LLM input | [Website](https://www.firecrawl.dev) · [GitHub](https://github.com/mendableai/firecrawl) |
| **Jina AI** | Embedding and reranking models for search and RAG | [Website](https://jina.ai) |

## MCP Servers & Tools

Model Context Protocol for connecting AI applications to external systems.

| Tool | Description | Links |
|------|-------------|-------|
| **Model Context Protocol** | Open standard for connecting AI apps to data sources and tools | [Website](https://modelcontextprotocol.io) · [GitHub](https://github.com/modelcontextprotocol) |
| **MCP Servers** | Official and community MCP server implementations | [GitHub](https://github.com/modelcontextprotocol/servers) |
| **Nanobots** | Platform for turning MCP servers into rich, autonomous AI agents | [Website](https://www.nanobot.ai) |
| **Claude Desktop** | Desktop app with native MCP support | [Website](https://claude.ai/download) |

## Observability & Monitoring

Platforms for tracing, debugging, and monitoring AI applications.

| Tool | Description | Links |
|------|-------------|-------|
| **Langfuse** | Open-source LLM engineering platform with traces, evaluations, and prompt management | [Website](https://www.langfuse.com) · [GitHub](https://github.com/langfuse/langfuse) |
| **LangSmith** | LangChain's platform for debugging, testing, and monitoring LLM apps | [Website](https://smith.langchain.com) |
| **Arize AI** | LLM observability and agent evaluation platform | [Website](https://arize.com) · [Phoenix](https://github.com/Arize-ai/phoenix) |
| **Helicone** | AI gateway and LLM observability for routing and debugging | [Website](https://www.helicone.ai) · [GitHub](https://github.com/Helicone/helicone) |
| **Fiddler AI** | AI observability and security platform with drift detection | [Website](https://www.fiddler.ai) |
| **Comet ML** | ML observability for experiments, models, and production | [Website](https://www.comet.com) |
| **Braintrust** | End-to-end platform for building, testing, and monitoring AI apps | [Website](https://www.braintrust.dev) |
| **Weights & Biases** | ML experiment tracking and model management | [Website](https://wandb.ai) |

## Evaluation & Testing

Frameworks for measuring and testing AI application quality.

| Tool | Description | Links |
|------|-------------|-------|
| **RAGAS** | Library for evaluating RAG pipelines with context and response metrics | [Website](https://docs.ragas.io) · [GitHub](https://github.com/explodinggradients/ragas) |
| **DeepEval** | Open-source framework for LLM evaluation and testing | [Website](https://www.confident-ai.com) · [GitHub](https://github.com/confident-ai/deepeval) |
| **TruLens** | Evaluation and tracing for groundedness and relevance (Snowflake) | [Website](https://www.trulens.org) · [GitHub](https://github.com/truera/trulens) |
| **Promptfoo** | CLI and library for testing and evaluating LLM outputs | [Website](https://www.promptfoo.dev) · [GitHub](https://github.com/promptfoo/promptfoo) |
| **Giskard** | Testing framework for ML models with vulnerability scanning | [Website](https://www.giskard.ai) · [GitHub](https://github.com/Giskard-AI/giskard) |

## Guardrails & Safety

Frameworks for ensuring safe, policy-compliant AI outputs.

| Tool | Description | Links |
|------|-------------|-------|
| **OpenAI Moderation API** | Free API for detecting harmful content across categories like hate, violence, self-harm, and sexual content | [Docs](https://platform.openai.com/docs/guides/moderation) |
| **Guardrails AI** | Validators for toxic language, hallucinations, PII, and compliance | [Website](https://www.guardrailsai.com) · [GitHub](https://github.com/guardrails-ai/guardrails) |
| **NeMo Guardrails** | NVIDIA's toolkit for programmable safety controls using Colang | [GitHub](https://github.com/NVIDIA/NeMo-Guardrails) |
| **Lakera Guard** | Real-time protection against prompt injection and data leakage | [Website](https://www.lakera.ai) |
| **Rebuff** | Self-hardening prompt injection detector | [GitHub](https://github.com/protectai/rebuff) |
| **LLM Guard** | Security toolkit for LLM interactions | [Website](https://llm-guard.com) · [GitHub](https://github.com/protectai/llm-guard) |

## Infrastructure & Compute

Platforms for running AI workloads at scale.

| Tool | Description | Links |
|------|-------------|-------|
| **Ray** | Distributed computing framework for scaling Python and AI applications | [Website](https://www.ray.io) · [GitHub](https://github.com/ray-project/ray) |
| **Modal** | Serverless AI infrastructure with sub-second cold starts and instant autoscaling | [Website](https://modal.com) |
| **Anyscale** | Managed Ray platform for ML/AI workloads | [Website](https://www.anyscale.com) |
| **dstack** | Open-source GPU provisioning across clouds and Kubernetes | [Website](https://dstack.ai) · [GitHub](https://github.com/dstackai/dstack) |
| **vLLM** | High-throughput inference engine with PagedAttention | [Website](https://vllm.ai) · [GitHub](https://github.com/vllm-project/vllm) |
| **TensorRT-LLM** | NVIDIA's library for optimized LLM inference | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **Triton Inference Server** | NVIDIA's multi-framework model serving | [GitHub](https://github.com/triton-inference-server/server) |

## Deployment Platforms

Platforms to deploy and serve AI models.

| Tool | Description | Links |
|------|-------------|-------|
| **Replicate** | Run, fine-tune, and deploy AI models with one line of code | [Website](https://replicate.com) |
| **Together AI** | Full-stack AI cloud with inference, fine-tuning, and GPU infrastructure | [Website](https://www.together.ai) |
| **Hugging Face Inference** | Deploy models with GPU acceleration | [Website](https://huggingface.co/inference-endpoints) |
| **Baseten** | ML infrastructure for deploying and serving models | [Website](https://www.baseten.co) |
| **BentoML** | Framework for building and deploying AI applications | [Website](https://www.bentoml.com) · [GitHub](https://github.com/bentoml/BentoML) |
| **AWS SageMaker** | End-to-end ML platform on AWS | [Website](https://aws.amazon.com/sagemaker/) |
| **GCP Vertex AI** | Google Cloud's ML platform | [Website](https://cloud.google.com/vertex-ai) |
| **Azure ML** | Microsoft's ML platform | [Website](https://azure.microsoft.com/en-us/products/machine-learning) |

## Data Processing & ETL

Tools for extracting and processing data for AI applications.

| Tool | Description | Links |
|------|-------------|-------|
| **Unstructured** | Extract and transform data from 64+ file types | [Website](https://unstructured.io) · [GitHub](https://github.com/Unstructured-IO/unstructured) |
| **Apache Airflow** | Workflow orchestration for data pipelines | [Website](https://airflow.apache.org) · [GitHub](https://github.com/apache/airflow) |
| **Prefect** | Modern workflow orchestration | [Website](https://www.prefect.io) · [GitHub](https://github.com/PrefectHQ/prefect) |
| **Dagster** | Data orchestration platform | [Website](https://dagster.io) · [GitHub](https://github.com/dagster-io/dagster) |
| **dbt** | Data transformation in your warehouse | [Website](https://www.getdbt.com) · [GitHub](https://github.com/dbt-labs/dbt-core) |

## Feature Stores

Centralized repositories for ML features.

| Tool | Description | Links |
|------|-------------|-------|
| **Feast** | Open-source feature store for ML | [Website](https://feast.dev) · [GitHub](https://github.com/feast-dev/feast) |
| **Tecton** | Real-time feature platform for ML with streaming, batch, and real-time feature pipelines | [Website](https://www.tecton.ai) · [Docs](https://docs.tecton.ai) |
| **Featureform** | Virtual feature store | [Website](https://www.featureform.com) · [GitHub](https://github.com/featureform/featureform) |
| **Hopsworks** | Feature store with MLOps capabilities | [Website](https://www.hopsworks.ai) · [GitHub](https://github.com/logicalclocks/hopsworks) |

## Prompt Management

Tools for versioning, testing, and optimizing prompts.

| Tool | Description | Links |
|------|-------------|-------|
| **PromptLayer** | Engineering workbench for versioning, testing, and monitoring prompts | [Website](https://www.promptlayer.com) |
| **Langfuse** | Prompt management with versioning alongside observability | [Website](https://www.langfuse.com) |
| **Agenta** | Open-source LLMOps platform for prompt engineering | [Website](https://www.agenta.ai) · [GitHub](https://github.com/Agenta-AI/agenta) |
| **Pezzo** | Open-source LLMOps platform | [Website](https://pezzo.ai) · [GitHub](https://github.com/pezzolabs/pezzo) |

## Fine-tuning Platforms

Platforms for customizing models on your data.

| Tool | Description | Links |
|------|-------------|-------|
| **Weights & Biases** | Platform for training and fine-tuning foundation models | [Website](https://wandb.ai) |
| **Predibase** | Fine-tuning and serving open-source LLMs | [Website](https://predibase.com) |
| **Together AI** | Fine-tuning services for custom models | [Website](https://www.together.ai) |
| **Replicate** | Model fine-tuning with custom datasets | [Website](https://replicate.com) |
| **Hugging Face AutoTrain** | No-code training for custom models | [Website](https://huggingface.co/autotrain) |
| **OpenPipe** | Fine-tuning platform optimized for cost reduction | [Website](https://openpipe.ai) |

## Embedding Models

Models for converting data into vector representations.

| Provider | Description | Links |
|----------|-------------|-------|
| **OpenAI Embeddings** | text-embedding-3-small and text-embedding-3-large | [Docs](https://platform.openai.com/docs/guides/embeddings) |
| **Voyage AI** | Specialized embedding models for code, legal, and more | [Website](https://www.voyageai.com) |
| **Cohere Embed** | Multilingual embeddings for search and RAG | [Docs](https://cohere.com/embed) |
| **Jina AI** | Embedding and reranking models | [Website](https://jina.ai) |
| **Sentence Transformers** | Open-source sentence embeddings | [GitHub](https://github.com/UKPLab/sentence-transformers) |
| **BGE Models** | BAAI open-source embedding models | [Hugging Face](https://huggingface.co/BAAI) |
| **Nomic Embed** | Open-source embeddings with long context | [Website](https://www.nomic.ai) |

## Workflow Orchestration

Tools for managing complex AI pipelines.

| Tool | Description | Links |
|------|-------------|-------|
| **Prefect** | Modern workflow orchestration for data and ML | [Website](https://www.prefect.io) · [GitHub](https://github.com/PrefectHQ/prefect) |
| **Apache Airflow** | Programmatic workflow authoring and scheduling | [Website](https://airflow.apache.org) · [GitHub](https://github.com/apache/airflow) |
| **Dagster** | Data orchestration with asset-based approach | [Website](https://dagster.io) · [GitHub](https://github.com/dagster-io/dagster) |
| **Temporal** | Durable execution for distributed systems | [Website](https://temporal.io) · [GitHub](https://github.com/temporalio/temporal) |
| **n8n** | Fair-code workflow automation with AI agent capabilities and 400+ integrations | [Website](https://n8n.io) · [GitHub](https://github.com/n8n-io/n8n) |
| **Flyte** | ML and data workflow platform | [Website](https://flyte.org) · [GitHub](https://github.com/flyteorg/flyte) |
| **Argo Workflows** | Kubernetes-native workflow engine | [Website](https://argoproj.github.io/workflows/) · [GitHub](https://github.com/argoproj/argo-workflows) |

## Local LLM Tools

Tools for running LLMs locally.

| Tool | Description | Links |
|------|-------------|-------|
| **Ollama** | Run LLMs locally with easy model management | [Website](https://ollama.com) · [GitHub](https://github.com/ollama/ollama) |
| **LM Studio** | Desktop app to run LLMs locally on macOS, Windows, Linux | [Website](https://lmstudio.ai) |
| **llama.cpp** | Inference of Meta's LLaMA in pure C/C++ | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **GPT4All** | Open-source ecosystem for running LLMs locally | [Website](https://gpt4all.io) · [GitHub](https://github.com/nomic-ai/gpt4all) |
| **Jan** | Open-source ChatGPT alternative that runs locally | [Website](https://jan.ai) · [GitHub](https://github.com/janhq/jan) |
| **LocalAI** | Drop-in OpenAI API replacement for local inference | [Website](https://localai.io) · [GitHub](https://github.com/mudler/LocalAI) |

## Code Context Tools

Tools for preparing codebases as context for LLMs.

| Tool | Description | Links |
|------|-------------|-------|
| **Repomix** | Pack entire repositories into a single AI-friendly file for use as LLM context | [Website](https://repomix.com) · [GitHub](https://github.com/yamadashy/repomix) |
| **OneFileLLM** | Convert codebases, documents, and web content into a single file for LLM input | [GitHub](https://github.com/jimmc414/onefilellm) |

## AI Coding Assistants

AI-powered code editors, IDEs, and development tools.

| Tool | Description | Links |
|------|-------------|-------|
| **Cursor** | AI-first code editor built for pair programming with AI | [Website](https://cursor.com) |
| **Cline** | Autonomous AI coding agent for VS Code that can create files, run commands, and use the browser | [GitHub](https://github.com/cline/cline) |
| **Roo Code** | AI-powered coding assistant for VS Code with multi-model support | [Website](https://roocode.com) · [GitHub](https://github.com/RooVetGit/Roo-Code) |

## Vibe Coding Tools

Build full applications from natural language descriptions.

| Tool | Description | Links |
|------|-------------|-------|
| **v0** | Generate UI components and full-stack apps from text prompts | [Website](https://v0.dev) |
| **bolt.new** | Build and deploy full-stack web apps from prompts in the browser | [Website](https://bolt.new) |
| **Lovable** | Build production-ready apps from natural language descriptions | [Website](https://lovable.dev) |
| **Base44** | Create apps by describing what you want in plain English | [Website](https://base44.com) |

## AI Specification Tools

Generate specifications and schemas using AI.

| Tool | Description | Links |
|------|-------------|-------|
| **OpenSpec** | Generate OpenAPI specifications from natural language descriptions | [GitHub](https://github.com/Fission-AI/OpenSpec) |

---

## Contributing

Read the [contribution guidelines](CONTRIBUTING.md) first. PRs that don't follow the guidelines will be closed.

## License

CC0 1.0 Universal
