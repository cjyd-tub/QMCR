# QMCR

This repository provides the anonymous implementation of **QMCR (Query-Adaptive Multi-Relational Complementarity Reasoning)** for set-aware complementary cloud API discovery.

QMCR performs complementary API discovery with respect to an existing query API set by integrating behavioral collaboration, function co-occurrence, semantic complementarity, provider complementarity, and functional substitution relations.

---

## Repository Structure

```text
QMCR/
├── config/             # Configuration files for different datasets and experimental settings
├── datasets/           # Processed PWA/HGA data and relation data used in the experiments
├── model/              # Core implementation of QMCR
├── script/             # Auxiliary scripts for running experiments
├── example.sh          # Example commands for running the model
├── .gitattributes
├── .gitignore
└── README.md



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
