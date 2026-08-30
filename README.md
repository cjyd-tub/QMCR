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
