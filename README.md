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


LLM-prompt

SYSTEM_PROMPT = """You are an expert in analyzing API functional complementarity for mashup application development.

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

OUTPUT: Return ONLY a JSON object, nothing else."""

USER_PROMPT_TEMPLATE = """Analyze if these two APIs are semantically complementary:

[API 1] {api1_name}
{api1_desc}

[API 2] {api2_name}
{api2_desc}

Question: Can these two APIs be meaningfully combined in a real application?
- If YES (different functions that synergize): is_complementary = true
- If NO (similar functions OR no integration scenario): is_complementary = false

Return JSON only:
{{"is_complementary": true/false, "confidence": 0.0-1.0, "reason": "one sentence explanation"}}"""
