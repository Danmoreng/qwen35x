# CPU-Prefill: Tiled Attention und automatische Pfadauswahl, 6. September 2026

Implementierung: `8185b485`.

Der lange Prefill verwendet jetzt FP32-Tiled-Attention mit Online-Softmax,
SIMD-KV-Transposition und GQA-Panel-Sharing. H128/Q4-G32-DOT4-Gewichte und FP16-KV
bleiben unverändert. Der Decode-Kernel bleibt unverändert. Der bisherige
Attention-Kernel ist mit `--cpu-attention rows` weiterhin direkt vergleichbar.

## Gemessener Gesamtdurchsatz

AMD Ryzen 9 9955HX3D, Windows/MSVC Release, acht Threads, Affinität `0xffff`,
identische feste Token-IDs und identischer Checkpoint. Prefill ohne LM-Head.
Kein GPU-Einsatz. Ausschließlich `scripts/benchmark-inference-seq.ps1`, sequenziell,
ohne parallele Builds, Tests oder Konvertierungen während der Messungen.

| Prompt-Tokens | Bisher: Rows | Neu: Auto | Durchsatzgewinn |
|---:|---:|---:|---:|
| 512 | 2121.35 | 2236.41 | +5.4% |
| 1024 | 1932.05 | 2162.66 | +11.9% |
| 2048 | 1605.26 | 2065.76 | +28.7% |
| 4096 | 1187.20 | 1875.22 | +58.0% |
| 8192 | 756.05 | 1619.47 | +114.2% |
| 257 | 2055.83 | 2087.48 | +1.5% |
| 4097 | 1197.20 | 1895.59 | +58.3% |


Werte in Tokens/s. Je Länge A/B/B/A mit drei Messungen nach einem Warmup pro
Aufrufgruppe: sechs Samples je Pfad, kein Sample entfernt. Kontextkapazität
mindestens 8192, bei Generierung mindestens Prompt plus Fortsetzung.
Für 512 Tokens ist oben die zusätzliche Bestätigung mit **drei Warmups** gezeigt:
die erste Serie schwankte stärker (Rows-Median 1987.81, Auto 2192.33).
Beide Serien bleiben vollständig veröffentlicht. Die Bestätigung ergibt
2121.35 → 2236.41 Tokens/s, also +5.4%, statt einen größeren, unsicheren Gewinn
aus der ersten Serie herauszustellen.

Bei 4096 Tokens steigt der Durchsatz um rund 58%, bei 8192 um rund 114%.
Der Kontextabfall verschwindet nicht: exakte kausale Full Attention betrachtet
weiterhin quadratisch viele Query-Key-Paare. Die Verarbeitung dieser Paare ist
jetzt wesentlich effizienter; daraus folgt keine konstante Geschwindigkeit bei
beliebig langen Kontexten und kein bewiesenes Hardware-unabhängiges Optimum.

Decode nach 512 Prompt-Tokens, 128 feste Ausgabetokens: **126.17 → 125.46 Tokens/s**
im A/B/B/A-Median (-0.57%). Kein gemessener Decode-Gewinn. Gezählt werden 127
Decode-Forward-Aufrufe; die erste Vorhersage entsteht im Prefill. Der neue Pfad
betrifft den Prefill. Die vollständigen Decode-Samples zeigen die Streuung.

## Was implementiert wurde

- Eigenständige AVX2/FMA/F16C- und AVX-512-FP32-Translation-Units und Dispatch,
  unabhängig von den VNNI-Gewichtskernels. `--cpu-attention-isa` erlaubt einen
  isolierten Attention-ISA-Vergleich bei unverändertem Q4-Backend.
- K-Panels transponiert als `[D, Bkv]`, V als `[Bkv, D]`, FP32-Akkumulatoren und
  Online-Softmax pro Query. SIMD-Lanes über Keys beziehungsweise Ausgabefeatures;
  vier Queryzeilen und zwei unabhängige SIMD-Spalten halten acht Akkumulatoren.
- 8×8-SIMD-Transposition statt skalarer verstreuter Stores; vier GQA-Heads können
  ein Panel teilen. Jede Query behält eigene Maxima, Summen, Gates und Kausalgrenzen.
- Begrenzter, wiederverwendbarer Scratch pro Executor-Partition; keine globale
  FP32-KV-Kopie. Typisch 200 KiB je Teilnehmer bei Bq=16, Bkv=32 und GQA-Gruppe 4.
  Output-Bereiche sind disjunkt; alle Jobs enden vor Workspace-Wiederverwendung.
- Explizit null gesetzte maskierte Wahrscheinlichkeiten, auch bei bereits vollständig
  geschriebenem Chunk-KV. Kein `exp(-inf)` als Maskenersatz. Vollständig maskierte
  Zeilen werden vor der Maxima-Differenz behandelt. Tails bleiben kausal korrekt.
- Opt-in-Feinprofilierung, tatsächliche Kernelbezeichnungen sowie Position,
  Tokenzahl, Query-Key-Paare, Tilegrößen und Teilnehmer je Layer/Chunk.
- Prefix-Snapshots berücksichtigen Attention-Modus, ISA, Chunk- und Tileeinstellungen;
  eine Strategieänderung invalidiert den Zustand, ohne die Gewichte neu zu laden.
- Tokenlisten können aus Dateien kommen, damit 8192er-Tests nicht am Windows-Limit
  für die Kommandozeilenlänge scheitern. Der Benchmarkrunner legt die Datei vor dem
  Prozessstart an und entfernt sie nach dessen Ende.

## Automatische Auswahl und getestete Alternativen

`--cpu-attention auto` ist Standard. Bis 128 bisherige/eingehende Kontext-Tokens
und bei Reststücken unter vier Tokens bleibt der Rows-Kernel aktiv. Bei längeren
Prompts beträgt der äußere Standardchunk 128, sonst 64. Unterstützte SIMD-Pfade
mit D=256 verwenden danach Tiled Attention; andere Formen/ISAs fallen zurück.
KV-Tile 32, Query-Tile zunächst 16 für AVX-512 beziehungsweise 8 für AVX2. Wenn
zu wenig unabhängige GQA-Tasks vorhanden wären, verkleinert Auto das Query-Tile
bis 4 oder trennt die Heads. Es orientiert sich an der tatsächlichen Threadzahl.

Die Wahl ist eine aus den Messungen abgeleitete Heuristik, kein Laufzeit-Autotuner.
Die expliziten Schalter bleiben für andere CPUs und reproduzierbare Vergleiche:
`--cpu-prefill-chunk-size 0..2048` (0=auto),
`--cpu-attention-query-tile 0|4|8|16`, `--cpu-attention-kv-tile 32|64|128`,
`--cpu-attention-gqa`, `--cpu-attention-isa auto|avx2|avx512`.
Bei explizitem `tiled` wird GQA nur mit dem entsprechenden Schalter aktiviert;
`auto` wählt es selbst. Ein expliziter Chunkwert überschreibt die Chunkheuristik.

Die Entwicklung wurde schrittweise gemessen. Die erste naive Fassung war langsamer
und wurde ersetzt. Nach SIMD-Transposition lag der 4096er-Prefill bei etwa 1471,
mit GQA-Sharing bei 1692, mit Tile-/Chunk-Tuning bei 1831 Tokens/s. Die abschließende
Fassung mit acht unabhängigen Akkumulatoren und automatischer Auswahl liegt bei 1875.
Diese Zwischenstände sind keine isolierten Varianten derselben finalen Binärdatei.
Separat gemessene AVX2-Attention verbesserte den langen Prefill ebenfalls;
die finale Mikrostruktur erreichte im Entwicklungsvergleich rund 1727 Tokens/s.

**Große äußere Chunks waren hier kein Gewinn:** Im isolierten GQA/Bq8/Bkv64-Test
bei 4096 Tokens lagen 64/128/256/512/1024/2048er-Chunks bei rund
1677/1726/1716/1646/1559/1504 Tokens/s. Deshalb werden lange Prompts nicht
pauschal in 1000er-Batches verarbeitet. Explizite Werte bleiben testbar.

## Wo die Kontextkosten liegen

Separate Diagnose bei 4096 Tokens, identische äußere 64er-Chunks, je ein Lauf
mit Feinprofilierung. Diese Zeiten sind **nicht** die Throughput-Messungen oben.

| Stage, Wall-Time in ms | Rows | Auto |
|---|---:|---:|
| Full-QKV | 84.64 | 82.01 |
| Q/K-Norm, RoPE, KV-Store | 15.95 | 17.11 |
| Full Attention | 1494.89 | 392.83 |
| Full-O-Projektion | 38.33 | 37.37 |
| Lineare Eingangsprojektion | 396.07 | 379.53 |
| DeltaNet-Rekurrenz | 232.61 | 232.31 |


Die neue Full-Attention-Stufe ist erheblich kürzer; Projektionen und DeltaNet
werden damit relativ wichtiger. Die Rohdaten enthalten auch feste 64-Query-Chunks
bei Position 0, 512, 1024, 2048 und 4032. `attention_wall_ms` ist verstrichene
Wall-Time. `*_worker_ms` sind summierte verstrichene Worker-Intervalle, keine
Wall-Time und keine vom Betriebssystem gemessene CPU-Zeit. Sie dürfen nicht zu
Wall-Time addiert werden. Full-Output ist O-Projektion; bei linearer Attention
enthält Output zusätzlich Norm/Gate, Prepare enthält Conv und Q/K/Alpha/Beta.

## Korrektheit und numerische Unterschiede

Neun CTest-Suites bestehen. Der neue Kerneltest umfasst 872 Konfigurationen:
FP16/FP32, AVX2/AVX-512, GQA mit/ohne Panel-Sharing, Tails, nichtnull Positionen,
Null-/kleine Werte, extreme endliche Scores und einen 4032er-Präfix. Eine dichte
FP64-Referenz verwendet identisch FP16-gerundete K/V. Maximaler absoluter Fehler
9.40591e-6. Manipulierte zukünftige Cachewerte beeinflussen frühere Query-Ausgaben
nicht. Scratch-/Output-Guards, paralleler Executor und AddressSanitizer werden geprüft.

Der vollständige Modelltest prüft echte gebatchte Prefills, mehrere Decode-Schritte,
FP16→FP32→FP16, bitidentische gecachte/ungecachte Wiederholungen derselben Strategie
und Cache-Invalidierung beim Strategiewechsel. Bitidentität zwischen **verschiedenen**
Attention-Algorithmen wird nicht vorausgesetzt: andere FMA-/Summationsreihenfolge
und Online-Softmax ändern FP32-Rundungen. Folgende Q8-/FP16-Rundungen und Layer
können diese Unterschiede verstärken.

Drei synthetische, von der Performance-Tokenfolge getrennte Langkontext-Fälle:
Prosa, Zahlen/IDs und JSON mit Kontextabruf; insgesamt 74 teacher-forced Positionen
mit vollständigem Vokabular. Referenzen: eingefrorener Q4-Stand `822a453` und
llama.cpp mit BF16-Gewichten aus demselben Checkpoint und FP16-KV. Verglichen wird
nach echtem Prefill und anschließendem Decode. Kein Replay-Ersatz für den Prefill.

| Fall | Prompt / Fortsetzung | KL alt→neu, Mittel / Maximum | ΔNLL | Top-1-Wechsel | BF16→alt / BF16→neu, mittlere KL |
|---|---:|---:|---:|---:|---:|
| narrative | 1485 / 21 | 0.000423 / 0.001506 | +0.002774 | 0 | 0.131555 / 0.130885 |
| ids | 4278 / 26 | 0.000439 / 0.001100 | -0.005902 | 0 | 0.123987 / 0.124260 |
| structured_recall | 4222 / 27 | 0.000552 / 0.001213 | +0.002134 | 1 | 0.206872 / 0.210393 |


ΔNLL ist neu minus alt auf der fest vorgegebenen Fortsetzung; niedriger ist besser.
Die Unterschiede sind klein, aber nicht null: eine knappe Top-1-Entscheidung wechselt.
Die BF16-Abstände enthalten auch Unterschiede zwischen Engines; sie sind keine reine
Messung des Q4-Quantisierungsfehlers. Drei synthetische Fälle sind keine allgemeine
Qualitätsgarantie. KLD-Tails und NLL sind veröffentlicht, statt aus einem identischen
Greedy-Beispiel Bitidentität oder uneingeschränkte Qualitätsparität abzuleiten.

## Reproduzieren und Daten

```powershell
.\scripts\benchmark-cpu-prefill-tiled.ps1 -Executable build/qwen35x.exe -OutDir benchmarks/prefill-rerun -Variants 'rows/auto/0/8/64/64','auto/auto/0/0/32/0','auto/auto/0/0/32/0','rows/auto/0/8/64/64'
python scripts/summarize-cpu-prefill-tiled.py benchmarks/prefill-rerun --out-prefix docs/prefill-rerun
```

Variantenformat: Modus/Attention-ISA/GQA/Query-Tile/KV-Tile/äußerer-Chunk.
Für Feinprofilierung `-FineProfile`, für Decode `-Lengths 512 -OutputTokens 128`.
Qualitätsläufe über `scripts/evaluate-cpu-prefill-attention.py` benötigen zusätzlich
den optionalen llama-Adapter und die vorhandene Transformers-Tokenizer-Umgebung.
Logit-Capture-Läufe sind keine Benchmarks.

[84 Hauptsamples](cpu-prefill-tiled-2026-09-06-samples.csv), [Zusammenfassung](cpu-prefill-tiled-2026-09-06-summary.csv),
[Binär-/Modellhashes und Einstellungen](cpu-prefill-tiled-2026-09-06-metadata.json),
[512er-Bestätigung](cpu-prefill-tiled-2026-09-06-short-confirm-samples.csv),
[Decode-Samples](cpu-prefill-tiled-2026-09-06-decode-samples.csv),
[Entwicklungsversuche](cpu-prefill-tiled-2026-09-06-experiments.csv),
[Stage-Rohdaten](cpu-prefill-tiled-2026-09-06-stages.json), [Qualitätsmetriken](cpu-prefill-tiled-2026-09-06-quality.json).
Der September-5-Vergleich mit llama.cpp bleibt ein unveränderter historischer
Vergleich; dessen Werte werden nicht mit neuen Werten zu einem angeblich neu
gemessenen Enginevergleich vermischt.

H256 bleibt zurückgestellt. DeltaNet-WY und Multi-Request-Batching sind die im Review
separat beschriebenen Folgeprojekte und wurden in dieser Attention-Änderung nicht umgesetzt.
