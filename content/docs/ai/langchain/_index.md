---
bookCollapseSection: false
weight: 2
title: LangChain
date: 2025-12-06
tags:
  - AI
---
# LangChain：用于构建大语言模型应用的开源框架及核心模块解析

LangChain 是一个用于构建大语言模型（LLM）应用的开源框架，核心目标是**连接 LLM 与外部资源、简化端到端工作流**，其模块设计围绕“数据输入-模型交互-逻辑编排-输出落地”的全链路展开。以下是 LangChain 的核心模块及详细说明：

### 一、核心基础模块（Models）

负责集成各类 AI 模型，是所有功能的“动力源”，支持主流模型提供商的无缝对接。

#### 1. 语言模型（LLMs）

- **作用**：处理文本生成、理解等核心任务（非对话式）。

- **支持类型**：

    - 闭源模型：OpenAI（GPT-3.5/4）、Anthropic（Claude）、Google（PaLM 2）、AWS（Bedrock）等；

    - 开源模型：Llama 2、Falcon、Mistral、LlamaCpp（本地部署）等。

- **核心功能**：文本生成、摘要、翻译、逻辑推理等。

#### 2. 聊天模型（Chat Models）

- **作用**：专门处理对话式交互（基于“消息”而非纯文本输入）。

- **特点**：输入是 `ChatMessage` 列表（如 `HumanMessage`、`AIMessage`、`SystemMessage`），输出是结构化的对话响应。

- **支持**：OpenAI ChatGPT、Anthropic Claude 2、Google Gemini Chat、开源聊天模型（如 ChatGLM）等。

#### 3. 嵌入模型（Embedding Models）

- **作用**：将文本转换为数值向量（嵌入向量），用于语义检索、相似度计算等。

- **支持**：OpenAI Embeddings、Cohere Embeddings、Hugging Face 开源嵌入模型（如 BERT、Sentence-BERT）、本地部署的嵌入模型（如 all-MiniLM-L6-v2）。

#### 4. 工具模型（Tool-calling Models）

- **作用**：支持模型主动调用外部工具（如搜索、数据库查询），是“代理（Agent）”功能的基础。

- **代表**：OpenAI Function Calling、Anthropic Tool Use、Google Gemini Function Calling 等。

### 二、提示工程模块（Prompts）

负责管理与模型的“交互语言”，优化提示质量以提升模型输出效果，是连接用户需求与模型的桥梁。

#### 1. 提示模板（Prompt Templates）

- **作用**：标准化提示格式，支持动态插入变量（如用户输入、上下文数据）。

- **示例**：问答模板 `“基于以下上下文回答问题：{context}\n问题：{question}”`。

- **扩展功能**：

    - 少样本模板（Few-shot Templates）：插入示例提升模型效果；

    - 聊天模板（Chat Templates）：适配聊天模型的消息格式；

    - 动态模板：根据条件切换提示逻辑（如 `ConditionalPromptTemplate`）。

#### 2. 示例选择器（Example Selectors）

- **作用**：从示例库中动态选择最相关的样本插入提示，优化少样本学习效果。

- **常见类型**：基于相似度选择、基于长度限制选择、随机选择等。

#### 3. 提示序列化（Prompt Serialization）

- **作用**：将提示模板保存为文件（如 JSON、YAML），支持复用、共享和版本控制（配合 LangChain Hub）。

#### 4. 输出解析器（Output Parsers）

- **作用**：将模型的非结构化输出转换为结构化数据（如 JSON、类实例、列表），方便后续处理。

- **常见类型**：

    - `JsonOutputParser`：强制模型输出 JSON 格式；

    - `PydanticOutputParser`：基于 Pydantic 模型定义输出结构；

    - `ListOutputParser`：输出列表格式数据。

### 三、数据连接模块（Data Connection）

负责对接外部数据（文档、数据库、API 等），解决 LLM“知识过时”“无法访问私有数据”的问题，是构建知识增强型应用的核心。

#### 1. 文档加载器（Document Loaders）

- **作用**：从各类数据源加载数据并转换为统一的 `Document` 格式（包含文本内容和元数据）。

- **支持数据源**：

    - 本地文件：TXT、PDF、Word、Excel、Markdown 等；

    - 网络资源：网页（URL）、RSS 订阅、GitHub 仓库；

    - 数据库：SQL（MySQL、PostgreSQL）、MongoDB、Redis；

    - 云服务：S3、Google Drive、Notion、Slack、Confluence 等。

#### 2. 文档分割器（Text Splitters）

- **作用**：将长文档分割为短文本块（适配 LLM 上下文窗口限制），同时保留语义完整性。

- **常见策略**：

    - 按字符/单词长度分割（`RecursiveCharacterTextSplitter`，默认推荐）；

    - 按语义分割（`SentenceTransformersTokenTextSplitter`，基于嵌入相似度）；

    - 按结构分割（如 Markdown 标题、PDF 页面）。

#### 3. 向量存储（Vector Stores）

- **作用**：存储文档嵌入向量，支持高效的语义检索（如相似性查询）。

- **支持类型**：

    - 本地/轻量：Chroma（默认推荐）、FAISS、SQLite-Vector；

    - 企业级：Pinecone、Weaviate、Milvus、Qdrant、Redis Vector；

    - 云服务：AWS OpenSearch、Google Cloud Vertex AI Vector Search。

#### 4. 检索器（Retrievers）

- **作用**：封装向量存储的检索逻辑，提供统一的检索接口（如 `retrieve(query)`），支持复杂检索策略。

- **常见类型**：

    - 基础检索：相似性检索（`SimilarityRetriever`）；

    - 高级检索：混合检索（向量+关键词）、上下文感知检索、多轮检索（`MultiQueryRetriever`）；

    - 自定义检索：基于规则/数据库查询的检索器。

#### 5. 文档转换器（Document Transformers）

- **作用**：对加载的文档进行预处理（如清洗、格式转换、信息提取）。

- **示例**：去除冗余文本、提取 PDF 表格、转换 HTML 为纯文本、基于 LLM 提炼文档关键信息。

### 四、工作流编排模块（Chains）

将单个组件（模型、提示、检索器等）串联为有序工作流，实现复杂任务的自动化执行（核心模块之一）。

#### 1. 基础链（Simple Chains）

- **作用**：线性串联少量组件，如“提示模板 → LLM → 输出解析器”。

- **示例**：`LLMChain`（最基础，直接调用 LLM）、`ChatChain`（适配聊天模型）。

#### 2. 组合链（Composite Chains）

- **作用**：串联多个基础链，支持分支、并行或循环逻辑。

- **常见类型**：

    - `SequentialChain`：线性执行多个链（前一个链的输出作为后一个链的输入）；

    - `RouterChain`：根据输入动态选择执行哪个链（如“问答链”或“总结链”）；

    - `ParallelChain`：并行执行多个链，合并输出结果。

#### 3. 预构建链（Pre-built Chains）

- **作用**：针对常见场景封装现成链，开箱即用。

- **典型场景**：

    - 问答（QA）：`RetrievalQA`（检索增强问答，“检索器+LLM”）、`ConversationalRetrievalChain`（带对话记忆的问答）；

    - 总结：`StuffDocumentsChain`（单文档总结）、`MapReduceChain`（长文档分块总结+合并）；

    - 翻译：`TranslationChain`；

    - 代码生成：`CodeChain`。

#### 4. 链序列化（Chain Serialization）

- **作用**：将链保存为文件（JSON/YAML），支持复用、共享和部署（配合 LangServe）。

### 五、智能代理模块（Agents）

让模型具备“自主决策能力”：根据用户需求，自主选择工具、调用链、调整策略，无需人工预设固定工作流（高级功能）。

#### 1. 代理核心（Agent Cores）

- **作用**：定义代理的决策逻辑（如“如何选择工具”“何时终止任务”）。

- **常见类型**：

    - 基于 LLM 思维链（ReAct）：`ReActAgent`（通过自然语言推理选择工具）；

    - 基于工具调用：`OpenAIFunctionsAgent`（利用模型的工具调用能力，更稳定）；

    - 无工具代理：`ConversationalAgent`（仅依赖模型和记忆进行对话）。

#### 2. 工具（Tools）

- **作用**：代理可调用的外部能力集合，是代理与现实世界交互的接口。

- **内置工具**：

    - 搜索工具：Google Search、Bing Search、SerpAPI；

    - 数据库工具：SQL 查询（`SQLDatabaseToolkit`）、MongoDB 查询；

    - 文件工具：读取/写入文件、处理 CSV；

    - 代码工具：执行 Python 代码（`PythonREPLTool`）、计算；

    - 其他：发送邮件、调用 API、地图查询。

- **自定义工具**：通过 `@tool` 装饰器或继承 `BaseTool` 类快速定义。

#### 3. 工具包（Toolkits）

- **作用**：封装一组相关工具，适配特定场景（如数据库操作、数据分析）。

- **示例**：`SQLDatabaseToolkit`（包含查询、表结构查看等工具）、`DataAnalysisToolkit`（包含 Pandas 分析、可视化工具）。

### 六、上下文记忆模块（Memory）

用于保存对话历史或任务上下文，让链/代理具备“长期记忆”能力（如多轮对话中记住用户之前的提问）。

#### 1. 基础记忆（Base Memory）

- **作用**：存储简单的对话历史（如 `ChatMessage` 列表）。

- **示例**：`ConversationBufferMemory`（完整保存对话历史）、`ConversationSummaryMemory`（用 LLM 总结对话历史，节省上下文空间）。

#### 2. 高级记忆（Advanced Memory）

- **作用**：支持复杂上下文管理（如实体记忆、关键信息提取）。

- **常见类型**：

    - `ConversationBufferWindowMemory`（仅保存最近 N 轮对话）；

    - `ConversationEntityMemory`（提取对话中的实体及关系，如“用户姓名、偏好”）；

    - `VectorStoreMemory`（将对话历史存储在向量库中，支持语义检索）；

    - `RedisChatMessageHistory`（将记忆存储在 Redis 中，支持分布式部署）。

#### 3. 记忆集成

- 可直接集成到 `Chain` 或 `Agent` 中，通过 `memory` 参数指定，自动管理上下文。

### 七、部署与监控模块（Callbacks & Observability）

支持应用的部署、日志监控、调试优化，适配生产环境需求。

#### 1. 回调系统（Callbacks）

- **作用**：在链/代理执行过程中触发钩子函数，用于日志记录、监控、数据收集。

- **常见回调**：

    - 日志回调：记录输入、输出、执行时间；

    - 监控回调：集成 Prometheus、Grafana 监控性能；

    - 调试回调：输出链/代理的执行步骤（如“调用了哪个工具”“生成了什么提示”）。

#### 2. LangSmith（可观测性平台）

- **作用**：LangChain 官方提供的调试、监控、评估平台，支持：

    - 追踪链/代理的执行链路（可视化工作流）；

    - 记录输入输出、中间结果、错误信息；

    - 评估模型/链的性能（如准确率、响应时间）；

    - 版本控制（提示、链、代理的迭代管理）。

#### 3. LangServe（部署工具）

- **作用**：将链/代理快速部署为 REST API 服务，支持高并发、负载均衡。

- **特点**：基于 FastAPI 构建，支持自动生成 OpenAPI 文档，适配 Docker/K8s 部署。

#### 4. LangChain Hub（共享平台）

- **作用**：共享和复用提示模板、链、代理配置（如公开的 QA 链、总结模板），支持版本控制。

### 八、生态集成模块（Integrations）

LangChain 核心优势之一是丰富的第三方集成，覆盖数据存储、工具、云服务等，无需重复造轮子。

#### 1. 模型集成

- 覆盖主流 LLM 提供商（OpenAI、Anthropic、Google、AWS、Azure、阿里云、腾讯云等）。

#### 2. 数据存储集成

- 数据库：SQL（MySQL、PostgreSQL）、NoSQL（MongoDB、Redis）、向量库（Pinecone、Weaviate 等）；

- 文件存储：S3、Google Drive、Dropbox、本地文件系统。

#### 3. 工具集成

- 搜索：Google、Bing、SerpAPI、Tavily；

- 办公软件：Notion、Slack、Confluence、Microsoft 365；

- 开发工具：GitHub、GitLab、Docker、Kubernetes；

- 其他：支付接口（Stripe）、地图（Google Maps）、邮件（SMTP）。

#### 4. 框架集成

- Web 框架：FastAPI、Flask、Django；

- 数据科学：Pandas、NumPy、Matplotlib；

- 深度学习：Hugging Face Transformers、PyTorch、TensorFlow。

### 模块关系总结

LangChain 的模块遵循“**输入（数据连接）→ 处理（提示+模型）→ 编排（链/代理）→ 记忆（上下文）→ 输出（部署）** ”的逻辑闭环：

- 数据连接模块解决“数据从哪来”；

- 提示+模型模块解决“如何与 AI 交互”；

- 链+代理模块解决“如何自动化完成复杂任务”；

- 记忆模块解决“如何保留上下文”；

- 部署模块解决“如何落地到生产”。

通过这些模块的灵活组合，可快速构建问答系统、聊天机器人、数据分析工具、自动化办公流程等各类 LLM 应用。
> （注：文档部分内容可能由 AI 生成）