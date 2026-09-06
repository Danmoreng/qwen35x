[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$Executable='build-prefill-tiled/qwen35x.exe',
  [int[]]$Lengths=@(512,1024,2048,4096,8192),
  # mode/attention ISA/GQA/query tile/KV tile/outer chunk
  [string[]]$Variants=@('rows/auto/0/8/64/64','tiled/avx2/0/8/64/64','tiled/avx512/0/8/64/64','tiled/avx512/1/8/64/64'),
  [int]$Runs=3,[int]$WarmupRuns=1,[int]$Threads=8,
  [int]$OutputTokens=0,[switch]$FineProfile,
  [long]$Affinity=65535
)
$ErrorActionPreference='Stop'
if(Test-Path $OutDir){throw "Output directory already exists: $OutDir"}
New-Item -ItemType Directory -Path $OutDir | Out-Null
$model='models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h'
@{executable=$Executable;exe_sha256=(Get-FileHash $Executable).Hash;model_sha256=(Get-FileHash $model).Hash;
  git=(git rev-parse HEAD);lengths=$Lengths;variants=$Variants;runs=$Runs;warmups=$WarmupRuns;threads=$Threads;
  affinity=$Affinity;output_tokens=$OutputTokens;fine_profile=[bool]$FineProfile;kv='fp16'} | ConvertTo-Json -Depth 5 | Set-Content "$OutDir/metadata.json"
$proc=[Diagnostics.Process]::GetCurrentProcess();$saved=$proc.ProcessorAffinity
try {
  $proc.ProcessorAffinity=[IntPtr]$Affinity
  $fixture=(Get-Content configs/qwen3_5_0_8b_text_1024x512_tokens.csv -Raw).Trim().Split(',')
  $index=0
  foreach($length in $Lengths) {
    $prompt=@(for($i=0;$i -lt $length;++$i){$fixture[$i%$fixture.Count]}) -join ','
    foreach($variant in $Variants) {
      $parts=$variant.Split('/'); if($parts.Count -ne 6){throw "Invalid variant $variant"}
      $name="case$index-p$length-$($variant.Replace('/','-'))"; ++$index
      $settings=@{Executable=$Executable;Modes=@('cpu-h128');CpuQ4H128=$model;CpuThreads=$Threads;
        CpuAttention=$parts[0];CpuAttentionIsa=$parts[1];CpuAttentionGqa=($parts[2] -eq '1');
        CpuAttentionQueryTile=[int]$parts[3];CpuAttentionKvTile=[int]$parts[4];CpuPrefillChunkSize=[int]$parts[5];
        PromptMode='prompt-tokens';PromptTokensCsv=$prompt;Runs=$Runs;WarmupRuns=$WarmupRuns;
        MaxNewTokens=[Math]::Max(1,$OutputTokens);MaxContext=[Math]::Max(8192,$length+$OutputTokens);
        KeepProfiles=$true;ProfileDir="$OutDir/profiles";CsvOut="$OutDir/$name.csv";RunLabel=$name;ProfileCpuPrefill=[bool]$FineProfile}
      if($OutputTokens -eq 0){$settings.PrefillOnly=$true} else {
        $settings.ForcedOutputTokensCsv=(@(for($i=0;$i -lt $OutputTokens;++$i){$fixture[($length+$i)%$fixture.Count]}) -join ',')
      }
      & "$PSScriptRoot/benchmark-inference-seq.ps1" @settings *> "$OutDir/$name.log"
      $rows=@(Import-Csv "$OutDir/$name.csv")
      if($rows.Count -ne $Runs){throw "Incomplete case $name"}
      foreach($row in $rows){if([int]$row.prompt_tokens -ne $length -or [int]$row.generated_tokens -ne $OutputTokens){throw "Wrong token count $name"}}
      Write-Host "Completed $name"
    }
  }
} finally {$proc.ProcessorAffinity=$saved}
