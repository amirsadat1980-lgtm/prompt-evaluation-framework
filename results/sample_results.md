# Prompt Evaluation Results

| Rank | Prompt | Composite | Accuracy | Relevance | Clarity | Consistency | Instruction Following |
|---|---|---|---|---|---|---|---|
| 1 | v1_naive | 73% | 50% | 79% | 45% | 90% | 100% |
| 2 | v2_role_primed | 67% | 25% | 71% | 49% | 90% | 100% |
| 3 | v3_few_shot_strict | 64% | 17% | 54% | 52% | 100% | 100% |

## Per-test-case detail

### v1_naive

**Case `case_llm_latency`**

> Large language models answer questions by running inference across a distributed system of GPU servers. When traffic spikes, inference latency increases because requests queue up while GPUs are busy. Teams continuously monitor throughput and latency to decide when it's time to add more servers.

_accuracy: 50%, relevance: 88%, clarity: 42%, consistency: 100%, instruction following: 100%_

**Case `case_vector_rag`**

> The system first converts documents into vector embeddings, which are numeric representations that capture the meaning of the text. These embeddings are stored in a vector database that can quickly find the most similar documents to a given query. It also lets teams update the knowledge base without retraining the entire model.

_accuracy: 75%, relevance: 75%, clarity: 44%, consistency: 70%, instruction following: 100%_

**Case `case_prompt_engineering`**

> Prompt engineering is the practice of designing inputs that reliably steer a language model toward the response you actually want. A naive prompt like 'summarize this' often produces inconsistent results because the model has to guess at length, tone, and audience. Few-shot prompting goes a step further by including one or two example input-output pairs directly in the prompt, showing the model the exact pattern to follow.

_accuracy: 25%, relevance: 75%, clarity: 49%, consistency: 100%, instruction following: 100%_

### v2_role_primed

**Case `case_llm_latency`**

> Large language models answer questions by running inference across a many computers of GPU servers. When traffic spikes, wait time increases because requests queue up while GPUs are busy. Teams continuously monitor speed and latency to decide when it's time to add more servers.

_accuracy: 25%, relevance: 88%, clarity: 54%, consistency: 70%, instruction following: 100%_

**Case `case_vector_rag`**

> The system first converts documents into meaning codes, which are numeric representations that capture the meaning of the text. These meaning codes are stored in a vector database that can quickly find the most similar documents to a given query. It also lets teams update the knowledge base without retraining the entire model.

_accuracy: 50%, relevance: 62%, clarity: 47%, consistency: 100%, instruction following: 100%_

**Case `case_prompt_engineering`**

> Prompt engineering is the practice of designing inputs that reliably steer a language model toward the response you actually want. A naive prompt like 'summarize this' often produces inconsistent results because the model has to guess at length, tone, and audience.

_accuracy: 0%, relevance: 62%, clarity: 46%, consistency: 100%, instruction following: 100%_

### v3_few_shot_strict

**Case `case_llm_latency`**

> In short, when traffic spikes, wait time increases because requests queue up while GPUs are busy. Teams continuously monitor speed and latency to decide when it's time to add more servers.

_accuracy: 25%, relevance: 62%, clarity: 63%, consistency: 100%, instruction following: 100%_

**Case `case_vector_rag`**

> In short, the system first converts documents into meaning codes, which are numeric representations that capture the meaning of the text. These meaning codes are stored in a vector database that can quickly find the most similar documents to a given query.

_accuracy: 25%, relevance: 38%, clarity: 45%, consistency: 100%, instruction following: 100%_

**Case `case_prompt_engineering`**

> In short, prompt engineering is the practice of designing inputs that reliably steer a language model toward the response you actually want. A naive prompt like 'summarize this' often produces inconsistent results because the model has to guess at length, tone, and audience.

_accuracy: 0%, relevance: 62%, clarity: 47%, consistency: 100%, instruction following: 100%_
