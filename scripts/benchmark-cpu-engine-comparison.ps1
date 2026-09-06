[CmdletBinding()]
param(
    [string]$OutDir = 'benchmarks/cpu-final-2026-09-05',
    [string]$QwenExe = 'build/qwen35x.exe',
    [string]$LlamaExe = 'build-llama-fixed-cpu/bin/llama-fixed-cpu-bench.exe',
    [string]$QwenModel = 'models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h',
    [string]$LlamaModel = 'models/gguf/qwen3.5-0.8b-review-target-Q4_0.gguf',
    [int[]]$ThreadCounts = @(8, 12),
    [int[]]$PrefillLengths = @(512, 1024, 2048, 4096),
    [int[]]$OutputLengths = @(128, 256, 512, 1024),
    [int]$DecodePromptLength = 512,
    [int]$Runs = 3,
    [int]$WarmupRuns = 1,
    # Preserve the September 5 row-kernel comparison unless explicitly changed.
    [ValidateSet('rows','auto','tiled')][string]$CpuAttention = 'rows',
    [int]$CpuPrefillChunkSize = 64,
    [int64]$Affinity = 65535
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
$benchProcess = [Diagnostics.Process]::GetCurrentProcess()
$savedAffinity = $benchProcess.ProcessorAffinity
try {
    if (Test-Path $OutDir) { throw "Refusing to mix runs into existing directory: $OutDir" }
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    if ($Affinity -ne 0) { $benchProcess.ProcessorAffinity = [IntPtr]$Affinity }
    $fixture = @((Get-Content configs/qwen3_5_0_8b_text_1024x512_tokens.csv -Raw).Trim().Split(',') | ForEach-Object { [int]$_ })
    # Ordinary text tokens only in the continuation: never an EOS/control token.
    $continuation = @($fixture | Where-Object { $_ -lt 248000 })
    if ($fixture.Count -eq 0 -or $continuation.Count -eq 0) { throw 'Empty fixture' }
    $workloads = @()
    foreach ($length in $PrefillLengths) { $workloads += [pscustomobject]@{Name="pp$length";Prompt=$length;Output=0} }
    foreach ($length in $OutputLengths) { $workloads += [pscustomobject]@{Name="p${DecodePromptLength}-out$length";Prompt=$DecodePromptLength;Output=$length} }
    $metadata = [ordered]@{
        created_utc = [DateTime]::UtcNow.ToString('o'); engine_commit = (git rev-parse HEAD)
        llama_commit = (git -C third_party/reference/llama.cpp rev-parse HEAD)
        thread_counts = $ThreadCounts; affinity = $Affinity; runs = $Runs; warmups = $WarmupRuns
        cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
        qwen_model = $QwenModel; llama_model = $LlamaModel
        qwen_sha256 = (Get-FileHash $QwenModel).Hash; llama_sha256 = (Get-FileHash $LlamaModel).Hash
        qwen_exe_sha256 = (Get-FileHash $QwenExe).Hash; llama_exe_sha256 = (Get-FileHash $LlamaExe).Hash
        fixture_sha256 = (Get-FileHash configs/qwen3_5_0_8b_text_1024x512_tokens.csv).Hash
        kv_cache = 'fp16'; llama_flash_attention = 'on'; llama_batch = 2048; llama_ubatch = 512
        qwen_chunk = $CpuPrefillChunkSize; qwen_attention = $CpuAttention; max_context = 8192; workloads = $workloads
        semantics = 'Prefill-only emits no logits. Decode uses fixed continuation with full-vocabulary logits; N outputs require N-1 timed forwards, first prediction occurs in prefill. No sampling or speculative decoding.'
    }
    $metadata | ConvertTo-Json -Depth 8 | Set-Content "$OutDir/metadata.json"
    $caseIndex = 0
    foreach ($threadCount in $ThreadCounts) {
        foreach ($workload in $workloads) {
            $prompt = @(for ($i=0; $i -lt $workload.Prompt; ++$i) { $fixture[$i % $fixture.Count] }) -join ','
            $forced = if ($workload.Output -gt 0) { @(for ($i=0; $i -lt $workload.Output; ++$i) { $continuation[$i % $continuation.Count] }) -join ',' } else { '' }
            $engines = if (($caseIndex++ % 2) -eq 0) { @('qwen35x','llama') } else { @('llama','qwen35x') }
            foreach ($engine in $engines) {
                $caseName = "$engine-t$threadCount-$($workload.Name)"
                $settings = @{Executable=$QwenExe;Modes=@('cpu-h128');CpuQ4H128=$QwenModel;CpuThreads=$threadCount;CpuIsa='auto';CpuKvCache='fp16';PromptMode='prompt-tokens';PromptName=$workload.Name;PromptTokensCsv=$prompt;Runs=$Runs;WarmupRuns=$WarmupRuns;MaxNewTokens=[Math]::Max(1,$workload.Output);MaxContext=8192;CsvOut="$OutDir/$caseName.csv";RunLabel=$caseName;KeepProfiles=$true;ProfileDir="$OutDir/profiles"}
                if ($engine -eq 'llama') { $settings.Executable=$LlamaExe; $settings.Modes=@('cpu-llama-fixed'); $settings.CpuGguf=$LlamaModel; $settings.Remove('CpuQ4H128') }
                else { $settings.CpuAttention=$CpuAttention; $settings.CpuPrefillChunkSize=$CpuPrefillChunkSize }
                if ($workload.Output -eq 0) { $settings.PrefillOnly=$true } else { $settings.ForcedOutputTokensCsv=$forced }
                Write-Host "Starting $caseName"
                & "$PSScriptRoot/benchmark-inference-seq.ps1" @settings *> "$OutDir/$caseName.log"
                $rows = @(Import-Csv "$OutDir/$caseName.csv")
                if ($rows.Count -ne $Runs) { throw "Incomplete run: $caseName" }
                foreach ($row in $rows) {
                    if ([int]$row.prompt_tokens -ne $workload.Prompt -or [int]$row.generated_tokens -ne $workload.Output) { throw "Token-count mismatch: $caseName" }
                }
                Write-Host "Completed $caseName"
            }
        }
    }
} finally {
    $benchProcess.ProcessorAffinity = $savedAffinity
    Pop-Location
}
