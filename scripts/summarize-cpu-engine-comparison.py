#!/usr/bin/env python3
"""Validate and summarize the paired, fixed-token CPU comparison."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-prefix', type=Path,
                        default=Path('docs/cpu-engine-comparison-2026-09-05'))
    args = parser.parse_args()
    meta = json.loads((args.directory / 'metadata.json').read_text(encoding='utf-8-sig'))
    samples, comparisons = [], []
    for threads in meta['thread_counts']:
        for workload in meta['workloads']:
            medians = {}
            for engine in ('qwen35x', 'llama'):
                source = args.directory / f"{engine}-t{threads}-{workload['Name']}.csv"
                with source.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != meta['runs']:
                    raise ValueError(f'Incomplete measurements: {source}')
                values = []
                for row in rows:
                    prompt, output = int(row['prompt_tokens']), int(row['generated_tokens'])
                    if (prompt, output) != (workload['Prompt'], workload['Output']):
                        raise ValueError(f'Wrong token counts: {source}')
                    profile = json.loads(Path(row['profile_json']).read_text(encoding='utf-8-sig'))
                    steps = max(0, output - 1)
                    actual_steps = profile.get('decode_forward_steps')
                    if actual_steps is None:
                        actual_steps = int(profile['forward_pass_tokens']) - prompt
                    if int(actual_steps) != steps:
                        raise ValueError(f'Wrong forward count: {source}: {actual_steps} != {steps}')
                    prefill_ms, decode_ms = float(row['prefill_time_ms']), float(row['decode_time_ms'])
                    tps = prompt * 1000 / prefill_ms if output == 0 else steps * 1000 / decode_ms
                    if not math.isfinite(tps) or tps <= 0:
                        raise ValueError(f'Invalid timing: {source}')
                    values.append(tps)
                    samples.append(dict(timestamp_utc=row['timestamp_utc'], engine=engine, threads=threads,
                        workload=workload['Name'], run=int(row['run_index']), prompt_tokens=prompt,
                        output_tokens=output, decode_forward_steps=steps,
                        load_time_ms=float(row['load_time_ms']), prefill_time_ms=prefill_ms,
                        decode_time_ms=decode_ms, measured_tokens_per_second=tps))
                medians[engine] = statistics.median(values)
            comparisons.append(dict(workload=workload['Name'], threads=threads,
                prompt_tokens=workload['Prompt'], output_tokens=workload['Output'],
                qwen35x_tps=medians['qwen35x'], llama_cpp_tps=medians['llama'],
                qwen35x_speedup=medians['qwen35x']/medians['llama'],
                throughput_gain_percent=100*(medians['qwen35x']/medians['llama']-1)))
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in [('-samples.csv', samples), ('-summary.csv', comparisons)]:
        with Path(str(prefix)+suffix).open('w', encoding='utf-8', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    Path(str(prefix)+'-metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    text = f'''# CPU-Abschlussvergleich: Qwen35x und llama.cpp, 5. September 2026

Verglichen wurden Qwen35x `{meta['engine_commit'][:8]}` und der am Abend aktualisierte
llama.cpp-Stand [`{meta['llama_commit'][:8]}`](https://github.com/ggml-org/llama.cpp/commit/{meta['llama_commit']}).
Die Tabelle zeigt Mediane von drei Messungen nach einem vollständigen Aufwärmlauf.
Alle Messungen liefen sequenziell durch `scripts/benchmark-inference-seq.ps1`.

## Bedingungen

- AMD Ryzen 9 9955HX3D, 16 physische Kerne / 32 logische Prozessoren, Windows,
  MSVC Release. Beide Prozesse erhalten dieselbe Affinität `0x{meta['affinity']:x}`
  (logische Prozessoren 0–15) und jeweils 8 bzw. 12 Threads. Die Einschränkung
  übernimmt die kontrollierte Zuordnung der vorherigen Versuche; dies ist keine
  Aussage über das Optimum auf allen 32 logischen Prozessoren oder anderen CPUs.
- Beide Modelle stammen aus demselben lokalen BF16-Checkpoint. Qwen35x verwendet
  H128-G32/Q4 mit DOT4-Packing; llama.cpp verwendet frisches **Q4_0 mit --pure**.
  Keine MTP-/Draft-Gewichte, keine spekulative Ausführung. Unterschiedliche
  Quantisierungsrezepte bleiben ein Unterschied; Q4_K_M/IQ4 wurden nicht gemessen.
- Nur CPU. llama.cpp: GGML_NATIVE, AVX-512/VNNI und AVX-VNNI, Repacking und
  Flash Attention aktiv, Batch 2048 / Microbatch 512. Qwen35x: automatische
  ISA-Auswahl, EVEX-VNNI/DOT4, Prefill-Chunks 64. Jede Engine verwendet ihre
  eigene bestehende Batch-Ausführung, statt beide auf denselben Kernel zu zwingen.
- Beide: FP16-KV, FP32-Rekurrenzzustand, reservierter Kontext 8192, Batchgröße
  einer Sequenz. Lade-/Kontextaufbauzeiten sind getrennt erfasst und nicht in
  den hier gezeigten Prefill-/Decode-Zeiten enthalten.
- Identische Token-IDs aus `configs/qwen3_5_0_8b_text_1024x512_tokens.csv`.
  Für längere Prompts wird die Datei zyklisch wiederholt. Die feste Fortsetzung
  verwendet deren normale Texttokens ohne Kontrolltokens. Keine Tokenisierung,
  keine zufällige Sampling-Arbeit, keine unterschiedlichen EOS-Abbrüche.
- Prefill wird separat **ohne LM-Head-Ausgabe** gemessen. Decode beginnt nach
  einem festen 512-Token-Prompt; jeder gemessene Decode-Schritt berechnet die
  vollständigen Vokabular-Logits. Der kleine llama-Adapter verwendet unveränderte
  Upstream-Kernel und die öffentliche API. Es ist kein unveränderter
  `llama-bench`-Zufallstokenlauf: dessen Eingaben und Prefill-Logits-Arbeit würden
  nicht exakt zu unserem Lauf passen.
- N Ausgabetokens entsprechen N−1 gemessenen Decode-Forward-Pässen, weil die
  erste Vorhersage im Prefill entsteht. Die Auswertung überprüft die tatsächlichen
  Forward-Zähler und verwendet **(N−1)/Decode-Zeit**. Die älteren Durchsatzwerte
  im Optimierungsbericht verwendeten noch N/Decode-Zeit.
- Pro Fall ein ungemessener Lauf und drei gemessene neue Prozesse, identisch für
  beide Engines. Die Reihenfolge der Engines wechselt zwischen den Fällen.
  Während der Messungen laufen keine Builds, Konvertierungen oder anderen Benchmarks.
  Einzelne Werte sind keine statistische Garantie für andere Last-/Temperaturzustände.

## Ergebnisse

Alle Durchsätze in Tokens/s; Prozentangaben beziehen sich auf den Durchsatz,
positiv bedeutet einen Vorteil für Qwen35x gegenüber llama.cpp.
'''
    for output_mode, title in [(False, 'Prefill'), (True, 'Decode nach 512 Prompttokens')]:
        text += f'\n### {title}\n\n| Tokens | Threads | Qwen35x | llama.cpp | Vorteil |\n|---:|---:|---:|---:|---:|\n'
        selected=[r for r in comparisons if bool(r['output_tokens']) == output_mode]
        selected.sort(key=lambda r: (r['output_tokens'] if output_mode else r['prompt_tokens'], r['threads']))
        for row in selected:
            length=row['output_tokens'] if output_mode else row['prompt_tokens']
            text += f"| {length} | {row['threads']} | {row['qwen35x_tps']:.2f} | {row['llama_cpp_tps']:.2f} | {row['throughput_gain_percent']:+.1f}% |\n"
    text += f'''
## Reproduktion und Nachweise

`./scripts/benchmark-cpu-engine-comparison.ps1 -OutDir <neues-Verzeichnis>` startet
alle Fälle. Vorhandene Ergebnisverzeichnisse werden nicht überschrieben. Die
Auswertung erfolgt mit `python scripts/summarize-cpu-engine-comparison.py <Verzeichnis>`.

Den Adapter und die offiziellen llama-Werkzeuge baut das CMake-Projekt unter
`scripts/bench/llama-fixed-cpu`; Build/Modell-Konvertierung siehe dessen README.
Die Submodul-Revision ist im Repository festgehalten. Modell-/Executable-/Fixture-
SHA256, CPU, Affinität und alle Fälle stehen in der begleitenden Metadaten-JSON.
Die begleitende Samples-CSV enthält alle 96 gemessenen Einzelläufe; zusätzliche
Rohprofile und Logs liegen lokal im in der Orchestrierung angegebenen Verzeichnis.

Alle acht CTest-Suiten, die drei Python-Tests für den Logitvergleich und die
persistenten FP16/FP32-/Prefix-Replay-Modelltests bestanden mit dem finalen Build.
Diese Messreihe prüft Geschwindigkeit, nicht Modellqualität. Sie beweist keine
allgemeine Qualitätsüberlegenheit von H128 über alle llama.cpp-Q4-Varianten.
H256 und weitere Optimierungsversuche wurden auf Wunsch zurückgestellt.
'''
    Path(str(prefix)+'.md').write_text(text, encoding='utf-8')
    for row in comparisons:
        print(f"{row['workload']:15s} t{row['threads']:2d}: {row['qwen35x_tps']:8.2f} / {row['llama_cpp_tps']:8.2f} tok/s  {row['throughput_gain_percent']:+6.1f}%")

if __name__ == '__main__':
    main()
