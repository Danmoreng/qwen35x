# CPU-Abschlussvergleich: Qwen35x und llama.cpp, 5. September 2026

Verglichen wurden Qwen35x `a760d229` und der am Abend aktualisierte
llama.cpp-Stand [`74a7c897`](https://github.com/ggml-org/llama.cpp/commit/74a7c897f049c17e7080423aa2111776eff6ebbf).
Die Tabelle zeigt Mediane von drei Messungen nach einem vollständigen Aufwärmlauf.
Alle Messungen liefen sequenziell durch `scripts/benchmark-inference-seq.ps1`.

## Bedingungen

- AMD Ryzen 9 9955HX3D, 16 physische Kerne / 32 logische Prozessoren, Windows,
  MSVC Release. Beide Prozesse erhalten dieselbe Affinität `0xffff`
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

### Prefill

| Tokens | Threads | Qwen35x | llama.cpp | Vorteil |
|---:|---:|---:|---:|---:|
| 512 | 8 | 2133.47 | 1101.73 | +93.6% |
| 512 | 12 | 1673.97 | 1028.32 | +62.8% |
| 1024 | 8 | 1938.11 | 1097.80 | +76.5% |
| 1024 | 12 | 1624.43 | 997.54 | +62.8% |
| 2048 | 8 | 1615.98 | 1050.80 | +53.8% |
| 2048 | 12 | 1414.79 | 958.22 | +47.6% |
| 4096 | 8 | 1190.83 | 1000.60 | +19.0% |
| 4096 | 12 | 1132.12 | 885.64 | +27.8% |

### Decode nach 512 Prompttokens

| Tokens | Threads | Qwen35x | llama.cpp | Vorteil |
|---:|---:|---:|---:|---:|
| 128 | 8 | 122.42 | 102.63 | +19.3% |
| 128 | 12 | 119.76 | 98.63 | +21.4% |
| 256 | 8 | 121.92 | 103.65 | +17.6% |
| 256 | 12 | 119.34 | 98.85 | +20.7% |
| 512 | 8 | 121.42 | 102.35 | +18.6% |
| 512 | 12 | 118.35 | 97.23 | +21.7% |
| 1024 | 8 | 118.81 | 101.33 | +17.3% |
| 1024 | 12 | 116.73 | 96.71 | +20.7% |

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
