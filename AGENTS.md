# Agent Notes

## Sandbox limitation

- `scripts/build.ps1` cannot be executed successfully in the default sandboxed environment for this repository.
- Building must be run with elevated permissions (outside sandbox restrictions).
- If a build/test run is needed, request escalation first and then execute the script.

## Benchmarking policy

- Do not run ad-hoc manual benchmark commands when measuring performance progress.
- Use `scripts/benchmark-inference-seq.ps1` for benchmark runs so execution is always sequential and results are written to CSV.
- Prefer comparable settings for progress tracking: `-Runs 3 -WarmupRuns 1 -MaxNewTokens 128 -MaxContext 256`.
