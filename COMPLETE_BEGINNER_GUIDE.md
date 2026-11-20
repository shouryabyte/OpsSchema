# OpsSchema: complete beginner guide

**Read this first.** OpsSchema is Shourya's independent assignment: it turns an everyday IT support request into strict JSON, searches the NASA Systems Engineering Handbook, and compares two compact Qwen model files. It is a learning/research project, not a production helpdesk. Results marked **PENDING** do not exist yet and must never be presented as measured.

## 1. The assignment in plain English

The company is testing whether you can build, measure, and honestly explain an AI system. Think of the base model as a student who has read many books. Fine-tuning is extra practice in writing support-ticket labels. RAG is letting the student search the NASA book before answering. Quantization is putting the same student’s knowledge into a smaller backpack.

| Task | What we built | Real input → output |
|---|---|---|
| 1 | Qwen/LoRA training pipeline | `VPN gateway drops sessions` → JSON with category, priority, action, system, summary |
| 2 | NASA PDF semantic retrieval | question → five page-numbered NASA passages |
| 3 | GGUF benchmark harness | same ten prompts → raw outputs, elapsed time, llama.cpp logs |

Why this problem: support teams receive messy text but downstream workflows need predictable fields. A wrong priority or invalid JSON can delay work. NASA systems engineering is a credible technical corpus for grounded answers.

## 2. Architecture and folders

```text
Task 1: request → prompt/tokenizer → Qwen + LoRA adapter → JSON parser → held-out metrics
Task 2: NASA PDF → PyMuPDF text → token chunks → MiniLM vectors → NumPy cosine search → cited evidence
Task 3: same prompts → Q4_K_M/Q8_0 GGUF → llama.cpp → raw logs/results
```

| Path | What it does | Show it in interview? |
|---|---|---|
| `task1_finetuning/src/data.py` | Makes deterministic 450/75/75 support records and validates allowed labels. | Yes: explain split separation. |
| `task1_finetuning/src/train.py` | Loads Qwen, applies rank-8 LoRA to `q_proj`/`v_proj`, runs CUDA QLoRA. | Yes: point to the CUDA guard and saved adapter. |
| `task1_finetuning/src/evaluate.py` | Strict JSON parser and controlled-field scoring helpers. | Yes. |
| `task1_finetuning/configs/train.json` | Records seed, learning rate, epochs, batch/accumulation and target modules. | Yes. |
| `task2_rag/src/rag.py` | PDF audit, cleaning, token windows, MiniLM embeddings, normalized dot-product search. | Yes. |
| `task2_rag/src/run_evaluation.py` | Refuses unapproved questions; evaluates both chunk settings. | Yes: integrity control. |
| `task3_optimization/scripts/benchmark.py` | Warmup plus three timed calls per prompt/quantization. | Yes. |
| `report/generate_report.py` | Refuses to create a PDF without actual evidence. | Yes. |

## 3. Run this project even if you have never coded before

1. Open Chrome, then open <https://colab.research.google.com/>.
2. Click **File → Upload notebook**. Upload `Shourya/colab/01_FINE_TUNING.ipynb` from your project folder.
3. Click **Runtime → Change runtime type → T4 GPU → Save**. A GPU is not guaranteed; if none appears, wait/retry later. Do not pretend a CPU run was training.
4. In the first code cell, set `REPO_URL` after you later make your own GitHub repository, or leave it blank and upload a ZIP of the `Shourya` folder when asked.
5. Run a cell by clicking the triangle at its left. Wait for the green tick. Red text means failure; read it before doing anything else.
6. Run notebooks in this order: `01_FINE_TUNING`, personally prepare questions, `02_RAG_EVALUATION`, `03_QUANTIZATION_BENCHMARK`, then `04_REPORT_GENERATION`.
7. Each notebook creates a ZIP or result directory. Download it before Colab disconnects, unzip it back into the matching `Shourya` folder on Windows.

A notebook is a click-to-run page. A `.py` file is the program it calls. PowerShell is a text-command window; only use it from the stated directory. Example Windows check:

```powershell
cd C:\Shourya\Tndrr\Shourya\task1_finetuning
python src\data.py
```

Expected: `train.jsonl`, `validation.jsonl`, and `test.jsonl` are regenerated. It does **not** train a model.

## 4. Task 1: fine-tuning from zero

Training means slightly changing selected model weights so the model learns this project’s response format. Testing means asking unseen examples, never examples it practiced. Validation helps choose settings during training; the untouched 75-row test set gives the fair final comparison.

Our label policy is business impact and requested work, not a single keyword. LoRA adds small trainable “adapter” matrices to attention projections; rank 8 controls adapter capacity and alpha 16 scales its effect. QLoRA keeps frozen base weights in 4-bit format to reduce VRAM. Learning rate `2e-4` is the step size; one epoch is one pass over training rows. Batch size 1 plus gradient accumulation 16 imitates a larger batch. Checkpointing trades extra computation for less VRAM.

**Example (measured fields are PENDING):**

```text
INPUT: Mumbai VPN gateway drops all remote sessions.
EXPECTED: {"category":"network","priority":"critical","action":"investigate",...}
BASE OUTPUT: PENDING real base-model generation
FINE-TUNED OUTPUT: PENDING real adapter generation
```

JSON validity asks “did it parse and have exactly five keys?” Category/priority/action accuracy compare each field with gold. Controlled exact match requires all three controlled fields correct. Exclude input tokens when reporting generation speed. A lower training loss alone does not prove better test accuracy; it can mean overfitting.

### Explain the 7B calculation

Open `task1_finetuning/MEMORY_CALCULATION.md`. Its rough assumptions are 7B parameters, bf16 (2 bytes), AdamW moments (8 bytes), optional fp32 master copy (4 bytes), and a separate activation allowance. Full fine-tuning approximately includes `weights=7B×2`, `weights+gradients=7B×(2+2)`, `Adam=7B×8`, master `7B×4`, plus activations. LoRA trains only ~20M assumed adapters; QLoRA estimates packed 4-bit base `7B×0.5` plus quantization metadata. This is a mathematical estimate, not `torch.cuda.memory_allocated()` and not hardware peak VRAM.

After the run, copy three rows from real saved predictions: input, gold, actual, and technical reason. Never call imagined examples failures.

## 5. Task 2: RAG and your 15 questions

RAG means **retrieve, then answer**. Instead of hoping the model remembers NASA, we extract its PDF with PyMuPDF, split each page into word windows, turn windows into embeddings, and retrieve the closest vectors. Normalized vectors make dot product equal cosine similarity. This project uses NumPy, not a vector-database framework.

NASA is audited in `task2_rag/results/document_audit.json`: 297 pages and SHA-256 are recorded. Compare 180-token/35-overlap and 320-token/60-overlap. Small chunks are precise; larger chunks preserve context. `recall@5` is the fraction of 15 questions whose exact verified evidence appears in at least one top-five retrieved chunk. BM25 is keyword matching; dense vectors help paraphrases; a cross-encoder reranker jointly reads query and passage after retrieval.

### How to create and approve my 15 RAG questions

1. Open the local NASA PDF and choose 15 distinct factual claims across relevant sections.
2. Write a question you can answer from one short passage.
3. Copy the exact passage, record the PDF page shown by your viewer, and use source ID `NASA-SP-2016-6105-REV2`.
4. Re-open the PDF, search the copied phrase, and mark it verified.
5. Fill the actual JSON below. Only when all 15 are personally verified, set `status` to `APPROVED`.

```json
{"id":"nasa-01","question":"MY QUESTION","exact_gold_passage":"EXACT COPY","page":0,"source_id":"NASA-SP-2016-6105-REV2","verification_status":"VERIFIED_BY_SHOURYA"}
```

This is a **format example**, not a valid question. The guard in `run_evaluation.py` must remain enabled. Run `python src/run_evaluation.py` from `task2_rag`; the resulting `results/evaluation.json` records top-five IDs/pages and summary recall. Record two real misses and say whether wording, chunk boundary, or embedding mismatch caused them.

## 6. Task 3: quantization

Local inference means running model weights with llama.cpp rather than a paid API. GGUF is its model-file format. Q4_K_M stores most weights at roughly four bits and usually uses less memory than Q8_0, which preserves more numeric detail. The verified Qwen files are `qwen2.5-0.5b-instruct-q4_k_m.gguf` and `qwen2.5-0.5b-instruct-q8_0.gguf` from the official Qwen GGUF repository.

The benchmark uses the same ten prompts, temp 0, 128-token cap, context 2048, one warmup pass, then three recorded repeats. Latency is seconds per request; throughput is generated tokens/second; RAM/VRAM are actual process/device observations, not estimates. Raw output lives in `task3_optimization/results/` after a real run.

### Explain serving 20 users

Imagine a restaurant: connected customers are not all ordering at the same second. Model weights are the kitchen; each actively generating request consumes KV-cache “notebook” space for prior tokens. The project estimate is in `CAPACITY_PLAN.md`: `2×layers×KV_heads×head_dim×bytes×tokens×live_sequences`, plus weights, runtime workspace, batching and safety headroom. Use continuous batching, queues, output/context limits and admission control. It is an estimate; CPU speed cannot prove a 24GB GPU rate.

## 7. Glossary: term — simple meaning | our example | why it matters

| Term | Meaning / OpsSchema example / importance |
|---|---|
| AI, ML, deep learning, neural network | Programs learn patterns; Qwen is a deep neural network; this is the assignment’s core. |
| LLM, open-weight, transformer | Text model whose weights can be downloaded; Qwen uses attention; enables adaptation. |
| token, tokenization, parameters | Text pieces, conversion to IDs, learned numbers; `train.py` tokenizes prompt/output. |
| attention, inference, prompt | Attention relates words; inference is asking; prompt instructs exact JSON. |
| training, fine-tuning, adapter | Practice, specialized practice, small added weights; LoRA trains adapters. |
| LoRA, QLoRA, rank, alpha | Low-rank adapters, 4-bit frozen base, capacity, scale; config is 8/16. |
| quantization, FP16, BF16, 4-bit, 8-bit | Fewer numeric bits; bf16 compute and Q4/Q8 compare memory/quality. |
| epoch, batch, accumulation, checkpointing | One pass, rows together, delayed update, recompute activations; config uses 3/1/16. |
| loss, overfitting, leakage | Training error, memorizing, test contamination; separate split IDs prevent leakage. |
| JSON schema, classification, accuracy, exact match | Allowed structure/labels/correct fraction/all controlled fields; evaluator measures these. |
| precision, recall, F1, macro-F1, confusion matrix | Error summaries; not yet used by this task, but report only if real predictions support them. |
| RAG, embedding, vector, cosine | Search-before-answer, numeric meaning, vector list, angle similarity; `rag.py` normalizes vectors. |
| chunk, overlap, NumPy, FAISS | Text window/shared words/math array/another vector index; this project uses NumPy. |
| BM25, reranker, recall@5 | Keyword rank/joint second rank/top-five evidence hit; compare chunk configurations. |
| hallucination, grounding, citation | Unsupported answer/evidence constraint/page reference; cite retrieved metadata only. |
| context window, temperature | Tokens model can see/randomness; benchmark fixes 2048/0. |
| GGUF, llama.cpp, Ollama | Local file/engine/alternative local runner; harness expects llama.cpp CLI. |
| CUDA, GPU, VRAM, RAM | NVIDIA compute/card memory/normal memory; Colab GPU required for training. |
| KV cache, latency, throughput | Stored attention state/delay/tokens per second; Task 3 measures or explains them. |
| continuous batching, concurrent users, memory optimization | Combine active work/live streams/use less memory; capacity plan discusses them. |

## 8. Troubleshooting

| Problem | Do this | Never do this |
|---|---|---|
| No GPU | Colab Runtime → Change runtime type → GPU; retry later. | Label CPU validation as training. |
| CUDA out of memory | Reduce max length/batch, keep accumulation; restart runtime. | Delete test rows or invent results. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt`; restart kernel. | Copy random packages into code. |
| Question guard stops | Finish all 15 exact evidence checks, then approve. | Change code to bypass guard. |
| PDF/model missing | Re-run audit/download and verify path/name. | Substitute a different model silently. |
| Report refuses | Read listed paths; run missing experiments. | Create fake JSON or a PDF by hand. |
| Colab disconnects | Download evidence ZIP after each task. | Assume temporary files are safe. |

## 9. Interview: a natural five-minute walkthrough

Open `README.md`, then `configs/train.json`. Say: “I built OpsSchema to turn messy support language into validated operational fields. I used Qwen with rank-8 LoRA because I needed a small adaptable model.” Open `data.py`: explain separate 450/75/75 splits. Open `rag.py`: “I independently extract NASA pages, chunk them, normalize MiniLM vectors, and preserve page IDs.” Open benchmark prompts and `CAPACITY_PLAN.md`. Fill only actual results: “On 75 held-out rows, base was **[read saved metric]** and adapter was **[read saved metric]**.” End: “My current limitation is **[real limitation]**; next I would improve **[real next step]**.”

### 50 interview drills (answer honestly; `WHERE` is where to show proof)

| # | Question | Simple truthful answer | Where |
|---:|---|---|---|
|1|What problem?|Support text to JSON.|README|
|2|Why Qwen?|Small instruction model.|train config|
|3|What is LoRA?|Small trained adapters.|train.py|
|4|Why QLoRA?|Less VRAM for frozen base.|train.py|
|5|Why rank 8?|Initial capacity choice, test it.|train.json|
|6|Why alpha 16?|Adapter scaling choice.|train.json|
|7|What is an epoch?|One pass over 450 rows.|train.json|
|8|Why held-out test?|Fair unseen measurement.|dataset|
|9|How prevent leakage?|Separate IDs/contexts.|data.py|
|10|What is JSON validity?|Parser plus exact keys.|evaluate.py|
|11|What is accuracy?|Correct fraction.|evaluate.py|
|12|Base result?|PENDING until saved.|results/|
|13|Adapter result?|PENDING until saved.|results/|
|14|Failure example?|Show real saved row only.|results/|
|15|Why NASA?|Systems-engineering source.|document audit|
|16|What is RAG?|Retrieve evidence first.|rag.py|
|17|What is a chunk?|Token window from a page.|rag.py|
|18|Why overlap?|Avoid boundary loss.|rag.py|
|19|Why two sizes?|Precision/context trade-off.|run_evaluation.py|
|20|What is an embedding?|Meaning vector.|rag.py|
|21|Why normalize?|Dot product becomes cosine.|rag.py|
|22|What is recall@5?|Gold evidence in five results.|evaluate.py|
|23|RAG score?|PENDING approval/run.|results/|
|24|BM25 vs dense?|Words vs meaning vectors.|guide|
|25|Reranker?|Second joint scoring stage.|guide|
|26|How cite?|Metadata, not generation.|rag.py|
|27|What is GGUF?|llama.cpp model file.|Task 3 README|
|28|Why Q4/Q8?|Memory/quality trade-off.|benchmark.py|
|29|Why identical prompts?|Fair comparison.|prompts.json|
|30|What is warmup?|Unrecorded first load/run.|benchmark.py|
|31|What is latency?|Wall-clock request time.|raw results|
|32|What is throughput?|Output tokens per second.|llama logs|
|33|Benchmark result?|PENDING real run.|results/|
|34|What is KV cache?|Stored attention state.|capacity plan|
|35|Why cache grows?|More tokens/sequences.|capacity plan|
|36|20 users possible?|Estimate only; queue streams.|capacity plan|
|37|CPU proves GPU speed?|No.|capacity plan|
|38|What does loss mean?|Training fit, not final proof.|trainer state|
|39|If accuracy drops?|Inspect split/format/overfit.|saved outputs|
|40|Invalid JSON?|Use parser, prompt, data review.|evaluate.py|
|41|Low recall?|Inspect exact spans/chunks.|evaluation JSON|
|42|Missed evidence?|Show real failure, propose rerank.|results|
|43|What did you implement?|Point to files you ran.|guide/files|
|44|What is reproducible?|Seed/config/raw artifacts.|train.json|
|45|Why no framework RAG?|Demonstrate components.|rag.py|
|46|Why not paid API?|Offline-capable retrieval.|requirements|
|47|Biggest limitation?|PENDING experiments/human review.|checklist|
|48|Next improvement?|Validated data/evaluator/reranker.|guide|
|49|Don’t know answer?|Say so, inspect evidence.|results|
|50|Demo order?|Task 1 → 2 → 3 → report.|colab/|

## 10. Final submission checklist

- [ ] Real training and base/adapter predictions saved
- [ ] 75 held-out examples evaluated; three real failures documented
- [ ] NASA audit and 15 personal verified questions complete
- [ ] Both retrieval configurations and two real misses saved
- [ ] Q4 and Q8 raw runs, memory/logs and ten-prompt quality review saved
- [ ] `python report/generate_report.py` succeeds; render/check readable PDF is ≤4 pages
- [ ] Create **one** GitHub repository only when ready; include code/configs/results/docs, exclude weights, caches, `.env`, PDFs if policy requires
- [ ] Submit through the stated form before 21 November 2026, 11:59 PM IST.

Mock practice: for a beginner, explain the first architecture diagram; for a technical interview, walk through normalized vectors and LoRA; for a live code interview, open `data.py`, alter a **documented experimental copy** of chunk size, then explain why results require rerunning. If asked something unknown: “I do not want to guess; I would check the saved artifact/config and report back.”
