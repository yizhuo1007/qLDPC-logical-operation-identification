# Identifying Logical Operations from Syndrome Histories for qLDPC Codes

This repository implements exact and scalable methods for identifying executed logical operations from syndrome histories and auxiliary measurement outcomes. It includes examples of automorphism sequences, logical Pauli measurements, T injections, and logical circuits on bivariate bicycle (BB) codes.

**Paper:** [*Identifying Logical Operations from Syndrome Histories for qLDPC Codes*](https://arxiv.org/abs/2610.40052)

## Identification Framework

Given candidate operations or circuits $h$ and a recovery-target mapping $\tau(h)=c$, the framework computes a prior-weighted likelihood for each target:

$$
Z_c(o) = \sum_{h:\tau(h)=c} \pi(h)P(o\mid h).
$$

The MAP decision selects the highest-scoring target. Recovery targets can represent individual operation sequences, ordered logical circuits, or their net logical actions. The automorphism example demonstrates net-action aggregation with `--target net-logical-action` and supports target-uniform priors through `--prior uniform_targets`. Scalable inference also supports approximate scores and likelihood bounds.

## Methods and Examples

| Method | Example | Inference |
| --- | --- | --- |
| Automorphism identification | `[[18,4,2]]`; 625 four-operation sequences | Exact syndrome likelihoods; logical-action aggregation |
| Logical Pauli measurement identification | `[[18,4,2]]`; 15 measurement candidates | Exact ideal and scalable noisy inference |
| T-injection identification | `[[18,4,2]]`; 6 measurement-axis candidates | Exact ideal and scalable noisy inference |
| Logical-circuit identification | `[[72,12,6]]` BV and `[[18,4,2]]` adaptive mixed circuits | Scalable circuit-level inference |

## Repository Structure

```text
├── src/qldpc_identification/
│   ├── models/       # BB codes and logical operations
│   ├── inference/    # Exact, scalable, and target-level inference
│   └── circuits/     # Logical-circuit inference interface
├── examples/         # Runnable examples and configurations
├── data/             # Saved measurement outcomes and model inputs
└── tests/            # Correctness and regression tests
```

## Quick Start

Requires **Python 3.12**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run one example for each method:

```bash
python examples/automorphism.py
python examples/pauli_measurement.py
python examples/t_injection.py
python examples/full_circuit.py --case bv-low-marker-scalable --record 0
```

Additional examples:

```bash
# Recover the net logical action of a four-operation sequence
python examples/automorphism.py --case q4-marker-noisy-exact --target net-logical-action

# Adaptive mixed logical circuit
python examples/full_circuit.py --case mixed-low-marker-scalable --record 0 --hypothesis I_ZI_IX
```

For noisy Pauli-measurement and T-injection examples, add `--case low-noisy-scalable --record 0 --hypothesis XI`. Use `--output` to save JSON results. Noisy scalable examples can take several minutes on a CPU.

### Logical-Circuit API

The circuit inference interface is also available in Python:

```python
from qldpc_identification.circuits import identify_circuit, load_circuit_case, load_observation

case = load_circuit_case("examples/configs/bv72.json")
outcomes = load_observation(case, 0)
result = identify_circuit(case, outcomes)
print(result["map_winners"])
```

## Population Evaluation

Evaluate saved noisy classical measurement outcomes and compare the resulting scores with reference labels:

```bash
python examples/reproduce_population.py --case pm --output outputs/pm_scores.csv
python examples/evaluate_scores.py --scores outputs/pm_scores.csv --truth data/code_18_4_2/pm_low_truth.csv --candidates 15
```

The population scripts also support T injection, BV, and mixed-circuit examples.

## Tests

```bash
python -m unittest discover -s tests -v
# Optional slower circuit-reference checks
PYTHONPATH=src python tests/slow_reference.py
```

## Citation

If you use this code or the identification methods, please cite the [paper](https://arxiv.org/abs/2610.40052):

```bibtex
@misc{tan2026identifying,
  title         = {Identifying Logical Operations from Syndrome Histories for qLDPC Codes},
  author        = {Tan, Yizhuo and Xu, Shifan and Ding, Yongshan and Szefer, Jakub},
  year          = {2026},
  eprint        = {2610.40052},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## License

Released under the [MIT License](LICENSE).
