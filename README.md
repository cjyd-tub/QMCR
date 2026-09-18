# QMCR

This repository provides the anonymous implementation of **QMCR (Query-Adaptive Multi-Relational Complementarity Reasoning)** for set-aware complementary cloud API discovery.

QMCR performs complementary API discovery with respect to an existing query API set by jointly modeling behavioral collaboration, function co-occurrence, semantic complementarity, provider complementarity, and functional substitution relations.

---

## Repository Structure

```text
QMCR/
├── config/             # Configuration files for different datasets and experimental settings
├── datasets/           # Processed PWA/HGA data and relation graphs used in the experiments
├── model/              # Core implementation of QMCR
├── script/             # Auxiliary scripts for running experiments
├── example.sh          # Example commands for running the model
├── .gitattributes
├── .gitignore
└── README.md
```

---

## Semantic Complementarity Relation Construction

The semantic complementarity relation is constructed offline before model training. It is used as one of the attribute-level relations in QMCR and is not involved in online LLM inference during complementary API discovery.

### Candidate Pair Selection

All unordered co-occurring API pairs with valid textual descriptions in the training-side composition evidence are treated as candidate pairs for semantic annotation.

The relation graphs are constructed only after the training, validation, and test partitions are fixed. Validation and test query-target facts are not used to construct semantic relation edges.

For each candidate pair, the LLM determines whether the two APIs are semantically complementary in practical application scenarios and returns:

- a binary complementarity decision;
- a confidence score in `[0, 1]`;
- a short explanation.

### LLM Settings

The semantic annotation process uses the following settings:

- **Model:** `gpt-3.5-turbo`
- **Temperature:** `0.3`
- **Top-p:** `0.9`
- **Maximum response length:** `150` tokens
- **Decoding seed:** not specified
- **Semantic-edge confidence threshold:** `0.7`

A semantic complementarity edge is retained only when:

1. `is_complementary = true`; and
2. `confidence >= 0.7`.

The semantic graph is constructed once offline and then used as a fixed relation graph in subsequent QMCR training and inference.

---

## Semantic Annotation Statistics

The semantic annotation process produces the following candidate-pair decisions and retained semantic-complementarity edges:

| Dataset | Candidate Pairs | Retained Semantic Edges | Retention Ratio |
|---|---:|---:|---:|
| PWA | 6,384 | 4,892 | 76.63% |
| HGA | 19,294 | 15,888 | 82.35% |

A serialization audit confirmed consistency between the accepted annotation records and the stored semantic relation graphs.

---

## Manual Validation

To assess the reliability of the retained semantic complementarity relations, a post-hoc manual validation was conducted by two annotators.

For each dataset, 100 retained semantic edges were randomly sampled and manually evaluated.

- **PWA:** 89/100 sampled edges were judged semantically complementary, corresponding to a retained-edge precision of **89%** with a Wilson 95% confidence interval of **81.4%–93.7%**.
- **HGA:** 92/100 sampled edges were judged semantically complementary, corresponding to a retained-edge precision of **92%** with a Wilson 95% confidence interval of **85.0%–95.9%**.

The manual validation was used only to assess semantic-relation quality and did not alter graph construction or model training.

---

## Offline Semantic Annotation Cost

The semantic complementarity graph is constructed once offline and is not involved in online recommendation.

The approximate annotation costs are:

| Dataset | Candidate-Pair Decisions | Time | Tokens |
|---|---:|---:|---:|
| PWA | 6,384 | 3 h 13 min | approximately 4.0M |
| HGA | 19,294 | 9 h 27 min | approximately 12.0M |

These offline annotation costs are reported separately from online inference latency.

---

## LLM Prompt

### System Prompt

```text
You are an expert in analyzing API functional complementarity for mashup application development.

TASK: Determine if two APIs are SEMANTICALLY COMPLEMENTARY based on their descriptions.

DEFINITION OF COMPLEMENTARY:
Two APIs are complementary when they provide DIFFERENT but SYNERGISTIC functionalities that developers would use TOGETHER in a single application.

POSITIVE EXAMPLES (is_complementary: true):
- Google Maps + Yelp: Location display + Business reviews (different functions, combined for location-based apps)
- Stripe Payment + SendGrid Email: Payment processing + Email notification (workflow integration)
- Twitter + Bit.ly: Social posting + URL shortening (content sharing workflow)
- Flickr + Facebook: Photo storage + Social sharing (media + social integration)
- Weather API + Calendar API: Weather data + Scheduling (travel/event planning)

NEGATIVE EXAMPLES (is_complementary: false):
- Google Maps + Bing Maps: Both are mapping services (SUBSTITUTES, not complementary)
- Stripe + PayPal: Both are payment processors (SUBSTITUTES)
- Random unrelated APIs with no integration scenario (NO SYNERGY)

KEY PRINCIPLE: Complementary means "different functions that work together", NOT "similar functions".

OUTPUT: Return ONLY a JSON object, nothing else.
```

### User Prompt Template

```text
Analyze if these two APIs are semantically complementary:

[API 1] {api1_name}
{api1_desc}

[API 2] {api2_name}
{api2_desc}

Question: Can these two APIs be meaningfully combined in a real application?
- If YES (different functions that synergize): is_complementary = true
- If NO (similar functions OR no integration scenario): is_complementary = false

Return JSON only:
{"is_complementary": true/false, "confidence": 0.0-1.0, "reason": "one sentence explanation"}
```

---

## Relation-Graph Construction Boundary

All relation-specific graphs used by QMCR are constructed exclusively from training-side evidence after the query-target partition is fixed.

The five relation types are:

- **CI:** co-invocation relation;
- **FC:** function co-occurrence relation;
- **SE:** semantic complementarity relation;
- **SP:** provider complementarity relation;
- **SU:** functional substitution relation.

Validation and test query-target facts are not used to generate relation edges.

This train-only graph-construction boundary is also applied to the corresponding relation-based and graph-based baselines.

---

## Reproducibility

For the main experiments, the implementation uses:

- **Python:** 3.9
- **Optimizer:** Adam
- **Learning rate:** `2e-4`
- **Batch size:** `32`
- **Batches per epoch:** `1024`
- **Training epochs:** `12`
- **Default discovery list length:** `K = 20`
- **Default neighborhood range:** `T = 6`
- **Main random seed:** `1024`

The main experiments use random seed `1024` for Python, NumPy, and PyTorch CPU/CUDA operations, with deterministic cuDNN execution enabled.

Repeated-run analyses use matched random seeds:

```text
42
52
62
```

under the same training budget and model-selection protocol.

---

## Running QMCR

Example commands for running QMCR are provided in:

```text
example.sh
```

Dataset-specific and experiment-specific settings are provided under:

```text
config/
```

The processed PWA/HGA data and relation data used in the experiments are provided under:

```text
datasets/
```
