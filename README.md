# qLDPC logical-operation identification

Compact Python examples for the identification methods in [the paper](https://arxiv.org/abs/2610.40052). The repository uses code parameters `[[18,4,2]]` and `[[72,12,6]]` throughout. It contains public observations and the model inputs needed to score them; hidden generating labels and execution records are not scorer inputs.

## Examples

| Task | Code and setting | Method | Candidates |
| --- | --- | --- | ---: |
| Automorphism words | `[[18,4,2]]`, four operations, known `Y0` marker, SEC after every operation | Exact ideal and noisy syndrome summation | 625 |
| Logical Pauli measurement | `[[18,4,2]]`, ideal and LOW noisy | Exact branch sum and count-aware scalable likelihood | 15 |
| T-injection measurement axis | `[[18,4,2]]`, ideal and LOW noisy | Exact branch sum and count-aware MP-JP likelihood | 6 |
| Bernstein–Vazirani circuit | `[[72,12,6]]`, LOW noise, known `Y0` marker | Local order-two fault factors with a 1024-state frontier | 4 |
| Mixed automorphism, T-injection, and Pauli circuit | `[[18,4,2]]`, LOW noise, marker, adaptive measurement basis | Full-circuit positive-sector likelihood | 8 |

The ideal examples evaluate a complete, finite branch space and use exact fractions. The noisy PM, T, and mixed examples score a retained probability mass and report a full-likelihood interval. A winner is certified only when its lower bound exceeds every competitor's upper bound. BV uses an approximate normalized score at each of three syndrome boundaries; its winner is not a likelihood certificate.

## Install and run

Python 3.12 is required.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The optional `PYTHONPATH=src python tests/slow_reference.py` checks frozen BV and mixed full-circuit scores; it can take several minutes.

Quick examples:

```sh
python examples/automorphism.py
python examples/automorphism.py --case q4-marker-noisy-exact --p-x 0.001 --p-z 0.001
python examples/pauli_measurement.py
python examples/pauli_measurement.py --case low-noisy-scalable --record 0 --hypothesis XI
python examples/t_injection.py
python examples/t_injection.py --case low-noisy-scalable --record 0 --hypothesis XI
python examples/full_circuit.py --case bv-low-marker-scalable --record 0
python examples/full_circuit.py --case mixed-low-marker-scalable --record 0 --hypothesis I_ZI_IX
```

Remove `--hypothesis` to evaluate every candidate for a noisy PM, T, or mixed record. Use `--record 0` through `--record 1023` to select a public noisy observation. Add `--output outputs/example.json` to save JSON. A complete population run requires scoring every public record under every candidate and may take substantial CPU time and memory; the quick commands above are numerical examples, not a rerun of the paper's full population statistics.

To stream a population run, use `python examples/reproduce_population.py --case pm --output outputs/pm_scores.csv`. The case may be `pm`, `t`, `bv`, or `mixed`; `--start`, `--stop`, and `--candidate` let you divide a long run into smaller jobs. The script caches each model within its candidate family and writes each score immediately. After combining complete candidate scores, `examples/evaluate_scores.py` compares them with the separate `*_truth.csv` labels. For example:

```sh
python examples/evaluate_scores.py --scores outputs/pm_scores.csv --truth data/code_18_4_2/pm_low_truth.csv --candidates 15
```

The truth CSVs are for evaluation only and are never read by the scorers. The population script emits an approximate normalized score for BV and retained likelihoods plus full bounds for PM, T, and mixed.

The noisy automorphism option sums independent X and Z faults exactly over the 14-bit syndrome quotient with perfect SEC after each operation. It illustrates Exact-Sum inference for this selected noise model; it does not reproduce the paper's circuit-level fault curves. `--p-x` and `--p-z` set the two fault rates. The ideal automorphism example reproduces unique identification of all 625 four-operation words.

## Files

```text
examples/                 One entry point for each task family
src/qldpc_identification/
  models/                 Code, logical ports, and ideal experiment models
  inference/              Exact and scalable scorers
data/code_18_4_2/         Ideal inputs, PM/T LOW factors, mixed models, records and labels
data/code_72_12_6/bv/     BV LOW MARKER circuits, codec, records and labels
tests/                    Quick exact and PM/T checks; optional slow circuit checks
```

The BV scorer reads only syndrome histories. The terminal Bernstein–Vazirani answer is not used to identify the circuit. The mixed scorer conditions each hypothesis on the basis selected by the observed prefix, scores the entire chronological record, and combines the positive magic-state sectors. All example scorers use public observations only.

The package is licensed under MIT; see `LICENSE`.
