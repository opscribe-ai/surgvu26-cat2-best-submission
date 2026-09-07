# Sweep the VLM's memory profile on a REAL Tesla T4 -- yours.
#
#   .\run_probe_t4.ps1                       # uses .\c131\input\...mp4
#   .\run_probe_t4.ps1 -Video path\to.mp4 -Image surgvu26-cat2:v61
#
# WHY THIS RUNS ON YOUR MACHINE. v6.1's NF4 checkpoint was "validated" on an
# H200 (sm_90, 139.8 GiB, peak 6.89 GiB) and then OOM'd on every graded case
# on the grader's T4. The difference is the ARCHITECTURE, not the size:
# PyTorch's flash SDPA backend needs sm_80+, and on sm_75 it falls back to a
# kernel that materialises the full [heads x tokens x tokens] score matrix --
# 3.0 GiB at 5184 tokens, held twice. A Quadro RTX 8000 (sm_75, 47 GiB) ran
# the identical image at a 16.5 GiB peak; a T4 has 14.56.
#
# CHTC has exactly one schedulable sm_75 card with enough memory and it is
# PI-owned -- 0 slots willing, ~14 h of backfill queue. Your T4 IS the
# grader's card, so this is not a simulation and there is nothing to argue
# with afterwards.
param(
  [string]$Image = "surgvu26-cat2:v61",
  [string]$Video = "$PWD\c131\input\endoscopic-robotic-surgery-video.mp4",
  [string]$Probe = "$PWD\probe_t4_memory.py",
  [string]$Question = "What type of forceps is mentioned?"
)

$ErrorActionPreference = "Stop"
foreach ($p in @($Video, $Probe)) {
  if (-not (Test-Path $p)) { throw "missing: $p" }
}
$Video = (Resolve-Path $Video).Path
$Probe = (Resolve-Path $Probe).Path
$out   = "probe_t4_results.jsonl"
Remove-Item $out -ErrorAction SilentlyContinue

# label, extra probe flags, PYTORCH_CUDA_ALLOC_CONF
$configs = @(
  @{ n = "sdpa 16x512 (5184 tok) -- the grader's exact plan, EXPECT OOM";
     a = @();                              e = "" },
  @{ n = "sdpa 16x512 + expandable_segments";
     a = @();                              e = "expandable_segments:True" },
  @{ n = "sdpa NO-MATH 16x512  -- can sm_75 use a non-materialising kernel?";
     a = @("--no-math-sdp");               e = "" },
  @{ n = "sdpa NO-MATH 16x512 + expandable";
     a = @("--no-math-sdp");               e = "expandable_segments:True" },
  @{ n = "sdpa 16x384 (2912 tok)";
     a = @("--size","384");                e = "expandable_segments:True" },
  @{ n = "sdpa 12x448 (2352 tok)";
     a = @("--frames","12","--size","448"); e = "expandable_segments:True" },
  @{ n = "sdpa 8x512  (2592 tok)";
     a = @("--frames","8");                e = "expandable_segments:True" }
)

foreach ($c in $configs) {
  Write-Host "`n--- $($c.n)" -ForegroundColor Cyan
  $args = @(
    "run","--rm","--gpus","all",
    "-v","${Video}:/data/case.mp4:ro",
    "-v","${Probe}:/tmp/probe.py",
    "-e","PYTORCH_CUDA_ALLOC_CONF=$($c.e)",
    "-e","PYTHONPATH=/opt/algorithm/src",
    "--entrypoint","python", $Image,
    "/tmp/probe.py","--video","/data/case.mp4","--question",$Question
  ) + $c.a
  # stderr carries the traceback; stdout carries the one JSON line we keep.
  $json = & docker @args 2>$null | Select-String -Pattern '^\{' | Select-Object -Last 1
  if ($json) { $json.Line | Tee-Object -Append -FilePath $out | Write-Host }
  else       { Write-Host "  (no JSON emitted -- rerun this one without 2>`$null to see why)" -ForegroundColor Yellow }
}

Write-Host "`n================ SUMMARY ================" -ForegroundColor Green
$rows = Get-Content $out | ForEach-Object { $_ | ConvertFrom-Json }
$rows | Format-Table status, tokens, peak_gib, cap_gib, bound, no_math_sdp,
                      alloc_conf, frames, size, wall_s -AutoSize
$fit = $rows | Where-Object { $_.status -eq "ok" } | Sort-Object tokens -Descending
if ($fit) {
  $b = $fit[0]
  Write-Host ("RICHEST PLAN THAT FITS YOUR T4: {0} tokens ({1}x{2}), peak {3} GiB, {4}{5}" -f `
    $b.tokens, $b.frames, $b.size, $b.peak_gib,
    $(if ($b.no_math_sdp) {"no-math-sdp"} else {"sdpa"}),
    $(if ($b.alloc_conf) {" +expandable"} else {""})) -ForegroundColor Green
} else {
  Write-Host "NOTHING FIT. Send me probe_t4_results.jsonl." -ForegroundColor Red
}
Write-Host "`nSend me $out"
