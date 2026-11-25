"""Question -> validated analysis plan.

The LLM proposes a typed plan over a fixed operator catalog; `validator` checks
every operator, column and parameter against the dataset schema; only the
deterministic operators compute numbers.
"""
