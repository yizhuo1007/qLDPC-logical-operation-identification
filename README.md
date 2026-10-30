# Identifying Logical Operations from Syndrome Histories for qLDPC Codes

Reference implementation and runnable Python examples for identifying executed logical operations and logical circuits from classical measurement outcomes.

- **Paper:** [*Identifying Logical Operations from Syndrome Histories for qLDPC Codes*](https://arxiv.org/abs/2610.40052)
- **Authors:** Yizhuo Tan, Shifan Xu, Yongshan Ding, and Jakub Szefer
- **Preprint:** arXiv:2610.40052 (2026)

> **Scope.** This is a lightweight method reference implementation, not a complete reproduction package for the paper's large-scale experiments.

## Overview

Classical measurement outcomes collected during quantum error correction can reveal information about which logical operations were executed. This repository demonstrates how to construct supported candidate models, evaluate the observed syndrome or auxiliary-measurement history under each candidate, and infer a logical operation or ordered logical circuit.

The release covers exact inference on selected finite models and scalable likelihood approximations on saved noisy observations. Its code and data are organized around two BB codes, `[[18,4,2]]` and `[[72,12,6]]`. The selected fault and observation models are documented below.

## Supported Examples

| Task | Code and setting | Method | Candidates |
| --- | --- | --- | ---: |
| Automorphism | `[[18,4,2]]`, four operations, known `Y0` marker, SEC after every operation | Exact ideal and noisy syndrome summation | 625 |
| Logical Pauli measurement | `[[18,4,2]]`, ideal and LOW noisy | Exact branch sum and count-aware scalable likelihood | 15 |
| T-injection measurement axis | `[[18,4,2]]`, ideal and LOW noisy | Exact branch sum and count-aware MP-JP likelihood | 6 |
| Bernstein–Vazirani circuit | `[[72,12,6]]`, LOW noise, known `Y0` marker | Local order-two fault factors with a 1024-state frontier | 4 |
| Mixed automorphism, T-injection, and Pauli circuit | `[[18,4,2]]`, LOW noise, marker, adaptive measurement basis | Full-circuit positive-sector likelihood | 8 |

The ideal PM and T examples evaluate a complete, finite branch space using exact fractions. The noisy PM, T, and mixed examples compute retained probability mass and report a full-likelihood interval. The BV example instead uses an approximate, uncalibrated normalized score at three syndrome boundaries and does not produce a likelihood certificate.

## Quick Start

### Requirements

- Python **3.12** (as specified in `pyproject.toml`)
- NumPy (installed automatically with the package)

The current automated checks run on Linux. Other operating systems have not yet been verified.

### Installation

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Run the Examples

```sh
# Exact ideal four-operation automorphism identification
python examples/automorphism.py

# Exact marginalization in a simplified noisy syndrome model
python examples/automorphism.py --case q4-marker-noisy-exact --p-x 0.001 --p-z 0.001

# Ideal 15-candidate logical Pauli measurement identification
python examples/pauli_measurement.py

# One candidate's noisy retained likelihood and bounds
python examples/pauli_measurement.py --case low-noisy-scalable --record 0 --hypothesis XI

# Ideal six-candidate T-injection identification
python examples/t_injection.py
python examples/t_injection.py --case low-noisy-scalable --record 0 --hypothesis XI

# Four-candidate Bernstein-Vazirani circuit identification
python examples/full_circuit.py --case bv-low-marker-scalable --record 0

# One hypothesis from the eight-candidate adaptive mixed-circuit example
python examples/full_circuit.py --case mixed-low-marker-scalable --record 0 --hypothesis I_ZI_IX
```

Remove `--hypothesis` to evaluate every candidate for a noisy PM, T, or mixed record. Use `--record 0` through `--record 1023` to select a public noisy observation. Add `--output outputs/example.json` to save JSON.

**Runtime note:** The ideal examples are quick. Noisy scalable scoring is substantially more expensive. In particular, scoring just one mixed-circuit hypothesis can take minutes on a CPU. Scoring all candidates or an entire 1,024-record population can take considerably longer. Start with the default ideal examples and the single-hypothesis commands above.

## How Identification Works

Every example follows the same overall workflow:

1. Load the supported code, circuit, and candidate-operation definitions.
2. Generate or load classical measurement observations.
3. Calculate candidate-dependent likelihoods or approximate scores using only the specified observation channels.
4. Return candidate scores, MAP recovery targets, and likelihood bounds where available.

The scripts call the actual implementations in `src/qldpc_identification/`. For more expensive examples, precompiled response models are included so visitors can run the inference without regenerating large physical models. The scripts do **not** simply look up precomputed final candidate decisions.

### Interpreting the Output

- **Automorphism versus logical action:** physical sequences and their induced logical actions are different possible recovery targets. The included four-operation example identifies physical sequences under its chosen marker and SEC schedule.
- **Ordered circuit versus net action:** an ordered circuit retains operation order, whereas a net action may identify multiple sequences as equivalent.
- **MAP recovery target:** for observation $o$, the paper scores target $c=\tau(h)$ by $Z_c(o)=\sum_{h:\tau(h)=c}\pi(h)P(o\mid h)$ and selects a target maximizing $Z_c(o)$. Strict unique identification requires the true target to be the sole maximizer. In these examples, each candidate is a distinct target with a uniform prior, so ranking candidate likelihoods gives the same decision.
- **Scalable decision:** replacing $P(o\mid h)$ with an approximate likelihood $\widetilde{P}(o\mid h)$ gives an approximate target score $\widetilde{Z}_c(o)$ and an approximate MAP decision.
- **Exact-model certification:** when valid bounds $Z_c^-\leq Z_c(o)\leq Z_c^+$ are available, target $c^*$ is certified as the exact MAP target if $Z_{c^*}^->\max_{c\ne c^*}Z_c^+$. The noisy PM, T, and mixed examples provide candidate likelihood bounds that induce these target bounds under their uniform priors.
- **BV score:** an approximate normalized score, not a certified likelihood or a conventional decoding success probability.

Correctly identifying the executed operation does not establish that the quantum computation itself was free of logical errors.

## Optional Population Evaluation

To stream a population run, use `python examples/reproduce_population.py --case pm --output outputs/pm_scores.csv`. The case may be `pm`, `t`, `bv`, or `mixed`; `--start`, `--stop`, and `--candidate` let you divide a long run into smaller jobs. The script caches each model within its candidate family and writes each score immediately. After combining complete candidate scores, `examples/evaluate_scores.py` compares them with the separate `*_truth.csv` labels. For example:

```sh
python examples/evaluate_scores.py --scores outputs/pm_scores.csv --truth data/code_18_4_2/pm_low_truth.csv --candidates 15
```

The truth CSVs are distributed for **evaluation only** and are never read by the scorers. The population script emits an approximate normalized score for BV and retained likelihoods plus full bounds for PM, T, and mixed. This workflow is optional and is not required to verify that the four example families run.

The noisy automorphism option sums independent X and Z faults exactly over the 14-bit syndrome quotient with perfect SEC after each operation. It illustrates Exact-Sum inference for this selected noise model; it does not reproduce the paper's circuit-level fault curves. `--p-x` and `--p-z` set the two fault rates. The ideal automorphism example reproduces unique identification of all 625 four-operation sequences.

## Repository Layout

```text
qLDPC-logical-operation-identification/
├── README.md
├── LICENSE
├── pyproject.toml
├── examples/
│   ├── automorphism.py
│   ├── pauli_measurement.py
│   ├── t_injection.py
│   ├── full_circuit.py
│   ├── reproduce_population.py
│   └── evaluate_scores.py
├── src/
│   └── qldpc_identification/
│       ├── codes.py
│       ├── models/
│       │   ├── automorphism.py
│       │   ├── pauli_measurement.py
│       │   └── t_injection.py
│       └── inference/
│           ├── exact.py
│           ├── noisy_automorphism.py
│           ├── scalable_*.py
│           └── _*.py
├── data/
│   ├── code_18_4_2/
│   │   ├── mixed_models/*.json.gz
│   │   ├── *.json
│   │   ├── *.csv
│   │   └── *.npz
│   ├── code_72_12_6/
│   │   └── bv/
│   │       ├── *.stim
│   │       ├── *.json
│   │       ├── *.csv
│   │       └── *.npz
│   └── reference_results.csv
├── tests/
│   ├── test_ideal.py
│   └── slow_reference.py
└── .github/
    └── workflows/
        └── tests.yml
```

## Tests

Run the quick checks after installation:

```sh
python -m unittest discover -s tests -v
```

The quick suite checks ideal BB/operation relations, exact likelihoods on selected finite instances, and frozen single-query PM/T scalable outputs. The optional slower suite checks reference outputs for BV and mixed circuits:

```sh
PYTHONPATH=src python tests/slow_reference.py
```

The slow suite can take several minutes. GitHub Actions currently runs the quick tests and selected automorphism, PM, and T commands on Ubuntu; it does not automatically execute every full-circuit path.

## Citation

If you use this implementation or build on the identification methods, please cite [the accompanying paper](https://arxiv.org/abs/2610.40052):

```bibtex
@misc{tan2026identifying,
  title        = {Identifying Logical Operations from Syndrome Histories for qLDPC Codes},
  author       = {Tan, Yizhuo and Xu, Shifan and Ding, Yongshan and Szefer, Jakub},
  year         = {2026},
  eprint       = {2610.40052},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph}
}
```

## License

This package is released under the [MIT License](LICENSE). Third-party dependencies retain their own licenses.

## Questions

Please open a GitHub issue for questions about the examples, installation, or methods.
