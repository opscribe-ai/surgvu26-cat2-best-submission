<#
.SYNOPSIS
    CHTC -> a Grand Challenge-ready image tarball, in one command.

.DESCRIPTION
    Pulls the staged build context, builds the image, and writes the gzipped
    `docker save` tarball GC accepts. Runs in Windows PowerShell 5.1 with no
    WSL distro and no MSYS tooling -- only `scp`, `tar` (both ship with
    Windows 10+) and Docker Desktop.

    WHY NOT `docker save ... | gzip`. That is the documented incantation and
    it is correct on Linux and macOS. In PowerShell it is a TRAP: there is no
    `gzip`, and the pipeline reencodes bytes as text, so even with one
    installed the archive comes out corrupt -- and corrupt in a way that
    uploads fine and fails on the far end.

    v6 PACKAGING (2026-08-30). v6 ships a CNN-ONLY image plus a SEPARATE model
    tarball, the same shape v5.2 used. This script produces the CONTAINER half
    only. The model half, `surgvu26-models-v6.tar.gz`, is downloaded from
    /staging as-is and uploaded to the MODEL slot; nothing here builds it.

.PARAMETER Version
    Image tag and output filename stem, e.g. v6.

.PARAMETER RemoteHost
    Where the context is staged. CHTC policy routes /staging transfers through
    transfer.chtc.wisc.edu rather than an access point.

.PARAMETER SkipPull
    Reuse a context tarball already sitting in -WorkDir.

.PARAMETER SkipBuild
    The image is already built under this tag; go straight to save and
    compress. Use when a build succeeded and a later step failed.

.PARAMETER WorkDir
    Where the context, the extracted tree and the output land. Defaults to
    %USERPROFILE%\surgvu-build.

.PARAMETER Clean
    Wipe every previous version from this machine before doing anything:
    prior extract trees, prior context tarballs, prior .tar/.tar.gz exports,
    and every local `surgvu26-cat2:*` Docker image. Then carry on with a
    normal run.

.PARAMETER CleanOnly
    Do the wipe and stop. Nothing is pulled, built or saved.

.PARAMETER DeepClean
    Widen the wipe from -WorkDir to a recursive scan of %USERPROFILE%. Older
    runs used a different -WorkDir (an earlier example in this file passed
    C:\Users\<you> directly), so v2/v3-era artefacts can be sitting outside
    the current one. This also removes stale MODEL tarballs -- including
    v5.2's surgvu26-models.tar.gz, which is the single most dangerous file to
    leave on this disk: it is five characters and 2 MB away from v6's, and
    uploading it silently scores v5.2. Everything removed is reproducible
    from /staging.

.PARAMETER ExpectMaxGB
    Hard ceiling on the COMPRESSED tarball, in GB. Default 6. A v6 CNN-only
    image compresses to roughly 1.3-1.6 GB; anything near 9 GB means VLM
    weights got baked into the image, which is the exact contamination that
    build 9714497 hit on the cluster. Exceeding this throws rather than
    printing a warning nobody reads at the end of a 20-minute run.

.EXAMPLE
    .\make_submission.ps1 -Version v6 -Clean

.EXAMPLE
    # wipe every older version off this machine and stop
    .\make_submission.ps1 -Version v6 -CleanOnly
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$RemoteHost = "nkalthoff@transfer.chtc.wisc.edu",
    [string]$RemotePath = "/staging/n/nkalthoff/surgvu26/submission_context.tar.gz",
    [string]$WorkDir    = "$env:USERPROFILE\surgvu-build",
    [switch]$SkipPull,
    [switch]$SkipBuild,
    [switch]$Clean,
    [switch]$CleanOnly,
    [switch]$DeepClean,
    [double]$ExpectMaxGB = 6.0
)

$ErrorActionPreference = "Stop"

function Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

# Every external command here reports failure through $LASTEXITCODE rather
# than by throwing, so $ErrorActionPreference does not cover them. Without
# this check a failed `docker build` would sail on and `docker save` would
# happily export the PREVIOUS version's image under the new tag -- the exact
# failure that produces a valid-looking upload of the wrong thing.
function Assert-LastExitCode($what) {
    if ($LASTEXITCODE -ne 0) {
        throw "$what failed with exit code $LASTEXITCODE"
    }
}

$image   = "surgvu26-cat2:$Version"
$context = Join-Path $WorkDir "submission_context.tar.gz"
$extract = Join-Path $WorkDir "context-$Version"
$rawTar  = Join-Path $WorkDir "surgvu26-cat2-$Version.tar"
$outTar  = "$rawTar.gz"

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# ---- 0. wipe previous versions ------------------------------------------
# WHY THIS EXISTS. Two of this project's worst near-misses were an artefact
# from an older version surviving into a newer run: a stale context producing
# a file labelled v3 holding v2, and a `docker save` exporting the previous
# tag. Both upload cleanly and score the wrong model. The cheapest defence is
# to leave nothing older on the disk to be picked up by accident.
if ($Clean -or $CleanOnly -or $DeepClean) {
    Step "Wiping previous versions from this machine"

    if ($DeepClean) {
        # Recursive, because older runs used a different -WorkDir. Slow the
        # first time; it is walking a user profile.
        $scanRoot = $env:USERPROFILE
        Write-Host "  deep scan under $scanRoot (this takes a minute)"
        $patterns = @("surgvu26-cat2-*.tar", "surgvu26-cat2-*.tar.gz",
                      "submission_context*.tar.gz", "surgvu26-models*.tar.gz")
        $hits = foreach ($p in $patterns) {
            Get-ChildItem -Path $scanRoot -Recurse -File -Filter $p -Force -ErrorAction SilentlyContinue
        }
        # NEVER delete THIS version's model tarball. `surgvu26-models*.tar.gz`
        # matches surgvu26-models-v6.tar.gz too, so running -DeepClean after
        # downloading the 9.9 GB model half would throw it away and the only
        # symptom would be a re-download.
        $keep = "surgvu26-models-$Version.tar.gz"
        $hits = $hits | Where-Object { $_.Name -ne $keep } | Sort-Object FullName -Unique
        if (-not $hits) {
            Write-Host "  no stray artefacts found outside $WorkDir"
        }
        foreach ($h in $hits) {
            Write-Host ("  rm file  {0}  ({1:N2} GB)" -f $h.FullName, ($h.Length / 1GB))
            Remove-Item -Force $h.FullName -ErrorAction SilentlyContinue
        }
        Get-ChildItem -Path $scanRoot -Recurse -Directory -Filter "context-v*" -Force -ErrorAction SilentlyContinue |
            ForEach-Object {
                Write-Host "  rm dir   $($_.FullName)"
                Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
            }
    }

    Get-ChildItem -Path $WorkDir -Directory -Filter "context-*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "  rm dir   $($_.FullName)"
            Remove-Item -Recurse -Force $_.FullName
        }

    Get-ChildItem -Path $WorkDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "surgvu26-cat2-*.tar" -or
                       $_.Name -like "surgvu26-cat2-*.tar.gz" -or
                       $_.Name -eq   "submission_context.tar.gz" } |
        ForEach-Object {
            Write-Host ("  rm file  {0}  ({1:N2} GB)" -f $_.FullName, ($_.Length / 1GB))
            Remove-Item -Force $_.FullName
        }

    # Docker images last: the tags are what `docker save` reads, so a stale
    # one here is the failure mode Assert-LastExitCode was written for.
    $tags = docker images --format "{{.Repository}}:{{.Tag}}" surgvu26-cat2 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "docker not reachable; skipped the image wipe"
    } elseif ($tags) {
        foreach ($t in $tags) {
            Write-Host "  rm image $t"
            docker rmi -f $t | Out-Null
        }
    } else {
        Write-Host "  no surgvu26-cat2 images present"
    }

    # Layer cache. Not strictly required for correctness -- layers are keyed
    # by content -- but a v5 build left ~10 GB of VLM layers behind and this
    # is the only thing that reclaims them.
    Step "Reclaiming Docker build cache"
    docker builder prune -af | Out-Null

    Write-Host ""
    Write-Host "Wipe complete." -ForegroundColor Green
    if ($CleanOnly) {
        Write-Host "-CleanOnly given; stopping here."
        return
    }
}

# ---- 1. pull -------------------------------------------------------------
if ($SkipPull -and $SkipBuild) {
    Step "Skipping pull and build; using the existing image $image"
    docker image inspect $image *> $null
    Assert-LastExitCode "docker image inspect (is $image built?)"
} elseif (-not $SkipPull) {
    Step "Pulling build context from $RemoteHost"
    scp "${RemoteHost}:${RemotePath}" $context
    Assert-LastExitCode "scp"
} else {
    Step "Reusing $context"
    if (-not (Test-Path $context)) { throw "no context at $context" }
}
if (-not ($SkipPull -and $SkipBuild)) {
    "{0:N1} MB" -f ((Get-Item $context).Length / 1MB) | Write-Host
}

# ---- 2. extract ----------------------------------------------------------
# Fresh directory each time. Extracting over a previous version would leave
# ITS models/ behind if a future context ever ships different filenames, and
# the build would bake in whichever the Dockerfile's COPY happened to match.
if (-not $SkipBuild) {
    Step "Extracting"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    tar -xzf $context -C $extract
    Assert-LastExitCode "tar"

    # WHICH MODEL IS IN HERE. -Version only names the tag and the output file;
    # it does NOT select a model. The context is a fixed path on /staging that
    # each CHTC build overwrites, so running with a bumped -Version against a
    # stale context produces a file labelled v6 holding v5 -- an upload that
    # looks entirely valid and silently scores the old model. Printing the
    # bound checkpoints makes that visible before the save and compress.
    Step "Checkpoints bound by this context"
    $cfg = Get-Content (Join-Path $extract "config\perception.json") -Raw | ConvertFrom-Json
    foreach ($role in $cfg.experts.PSObject.Properties.Name) {
        $e = $cfg.experts.$role
        "  {0,-6} {1,-28} {2}" -f $role, $e.checkpoint_name, $e.backbone | Write-Host
    }

    # CONTAMINATION TRIPWIRE, local half. v6's image is CNN-ONLY: the VLM
    # travels in the model tarball. `.safetensors` under the extracted tree
    # means VLM weights are about to be baked in, which is what cluster build
    # 9714497 did silently -- caught there only because the context came out
    # at 5.49 GB instead of ~230 MB. Same check, on this machine, before the
    # build rather than after the upload.
    Step "Verifying the context is CNN-only"
    $weights = Get-ChildItem -Path $extract -Recurse -File -Filter "*.safetensors" -ErrorAction SilentlyContinue
    if ($weights) {
        $weights | ForEach-Object { Write-Host "  $($_.FullName)" }
        throw ("Context carries {0} .safetensors file(s). v6 ships a CNN-only " +
               "image with the VLM in the model tarball, so this context is " +
               "not the one you want." -f $weights.Count)
    }
    $mdl = Join-Path $extract "models"
    Get-ChildItem -Path $mdl -File | ForEach-Object {
        "  {0,-28} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB) | Write-Host
    }
    Write-Host "  no .safetensors present -- CNN-only, as v6 expects" -ForegroundColor Green
}

# ---- 3. build ------------------------------------------------------------
# --platform linux/amd64 is not cosmetic: the GC validator compares
# config["architecture"] against linux/amd64 and rejects anything else.
# The build is self-gating -- it fails rather than producing a bad image if a
# checkpoint sha256 does not match config/perception.json, if the serving
# thresholds are not bound to those same weights, or if the entrypoint cannot
# answer a synthetic case offline on CPU.
if (-not $SkipBuild) {
    Step "Building $image"
    docker build --platform linux/amd64 -t $image $extract
    Assert-LastExitCode "docker build"
}

# ---- 4. save -------------------------------------------------------------
Step "Saving image (no progress output; several minutes is normal)"
if (Test-Path $rawTar) { Remove-Item -Force $rawTar }
if (Test-Path $outTar) { Remove-Item -Force $outTar }
docker save -o $rawTar $image
Assert-LastExitCode "docker save"

# CAPTURE THE SIZE NOW, not after compression. Both compression paths below
# DELETE the .tar on success (pigz and gzip both replace their input), so a
# later `(Get-Item $rawTar).Length` would throw -- and it would throw during
# the verify step, which is exactly where you least want a spurious failure.
[long]$rawBytes = (Get-Item $rawTar).Length
"{0:N2} GB uncompressed" -f ($rawBytes / 1GB) | Write-Host

# ---- 5. compress ---------------------------------------------------------
# REWRITTEN 2026-08-30. The previous implementation handed a MULTI-LINE
# here-string to `docker run ... alpine sh -c`, and that one command failed
# three separate times, each time AFTER the pull, the build and the multi-GB
# save -- the steps that cost real time:
#
#   2026-08-17  \$(nproc) used the SH escape. PowerShell has no \ escape, so
#               the guard did nothing and PowerShell ran `nproc` on Windows.
#   2026-08-26  the here-string emitted CRLF; sh does not treat \r as
#               whitespace, so `fi\r` is not `fi` and it died with
#               "pigz: line 1: syntax error: unexpected end of file".
#   2026-08-30  reported again by the user at the same point in the run.
#
# The pattern is the problem, not any one bug in it: a multi-line string
# written by PowerShell, quoted by PowerShell's native-argument layer, and
# parsed by sh has three independent chances to be mangled, and every failure
# lands after ~20 minutes of work.
#
# So compression is now .NET GZipStream, in-process. No container, no sh, no
# quoting, no line endings, no network, no `apk add`, and nothing to escape.
# It is single-threaded and therefore slower than pigz was -- but v6's image
# is CNN-only at ~3.5 GB rather than v5's ~12 GB, so this is a few minutes,
# and a few slow minutes that always work beat a fast path that has now
# failed three for three. Output is ordinary gzip; GC cannot tell the
# difference.
function Compress-Gzip($inPath, $outPath) {
    $in = $null; $out = $null; $gz = $null
    try {
        $in  = [System.IO.File]::OpenRead($inPath)
        $out = [System.IO.File]::Create($outPath)
        $gz  = New-Object System.IO.Compression.GZipStream(
                   $out, [System.IO.Compression.CompressionMode]::Compress)
        $in.CopyTo($gz, 4194304)
    } finally {
        if ($gz)  { $gz.Dispose() }
        if ($out) { $out.Dispose() }
        if ($in)  { $in.Dispose() }
    }
}

Step "Compressing (the longest step)"
$sw = [Diagnostics.Stopwatch]::StartNew()

# FAST PATH: pigz in a throwaway Alpine container, as ONE LINE. Same tool the
# old script used, minus the pattern that kept breaking it -- a single-line
# string cannot carry an embedded CRLF, contains no `$` for PowerShell to
# interpolate, and needs no escape characters from either language. If
# anything at all goes wrong (no network for `apk`, no daemon, a mangled
# mount) it is simply reported and the .NET path below runs instead.
$sh = "apk add --no-cache pigz >/dev/null 2>&1 && pigz -f 'surgvu26-cat2-$Version.tar' || gzip -f 'surgvu26-cat2-$Version.tar'"
Write-Host "  trying pigz (parallel)..."
docker run --rm -v "${WorkDir}:/w" -w /w alpine sh -c $sh
$fastOk = ($LASTEXITCODE -eq 0) -and (Test-Path $outTar)

if (-not $fastOk) {
    Write-Warning "parallel path did not produce $outTar; using in-process gzip"
    if (Test-Path $outTar) { Remove-Item -Force $outTar }
    if (-not (Test-Path $rawTar)) {
        throw "the .tar is gone but no .tar.gz was produced; rerun with -SkipPull -SkipBuild"
    }
    Compress-Gzip $rawTar $outTar
} else {
    Write-Host "  pigz succeeded"
}

$sw.Stop()
"{0:N1} min" -f $sw.Elapsed.TotalMinutes | Write-Host

# ---- 6. verify -----------------------------------------------------------
# A FULL decompress, not a header peek. This is the same guarantee `gzip -t`
# gave, and it is the check that stands between a truncated archive and an
# upload that fails on the far end after the deadline.
Step "Verifying (full decompress)"
$vin = $null; $vgz = $null
try {
    $vin = [System.IO.File]::OpenRead($outTar)
    $vgz = New-Object System.IO.Compression.GZipStream(
               $vin, [System.IO.Compression.CompressionMode]::Decompress)
    $buf = New-Object byte[] 4194304
    [long]$total = 0
    while (($n = $vgz.Read($buf, 0, $buf.Length)) -gt 0) { $total += $n }
} finally {
    if ($vgz) { $vgz.Dispose() }
    if ($vin) { $vin.Dispose() }
}
if ($total -ne $rawBytes) {
    throw "gzip verify: decompressed to $total bytes, expected $rawBytes"
}
"  decompressed to {0:N0} bytes, matches the save exactly" -f $total | Write-Host

# Both compression paths delete the .tar themselves; this only catches the
# case where one did not.
if (Test-Path $rawTar) { Remove-Item -Force $rawTar }

# ---- 7. report -----------------------------------------------------------
$sizeGB = (Get-Item $outTar).Length / 1GB
Write-Host ""
Write-Host "READY  $outTar" -ForegroundColor Green
Write-Host ("{0:N2} GB compressed" -f $sizeGB)

if ($sizeGB -gt $ExpectMaxGB) {
    throw ("Compressed image is {0:N2} GB, over the -ExpectMaxGB ceiling of " +
           "{1:N2} GB. A v6 CNN-only image should land near 1.5 GB; this size " +
           "means VLM weights were baked in. DO NOT UPLOAD IT. The file is on " +
           "disk if you want to inspect it." -f $sizeGB, $ExpectMaxGB)
}
if ($sizeGB -gt 10) {
    Write-Warning "Over the 10 GB Grand Challenge ceiling."
}

Write-Host ""
Write-Host "TWO UPLOADS. v6 is a CNN-only image PLUS a model tarball." -ForegroundColor Yellow
Write-Host ""
Write-Host "  CONTAINER slot   $outTar"
Write-Host "  MODEL slot       surgvu26-models-v6.tar.gz   (9.9 GB, download separately)"
Write-Host ""
Write-Host "    scp ${RemoteHost}:/staging/n/nkalthoff/surgvu26/surgvu26-models-v6.tar.gz ."
Write-Host ""
Write-Host "Leaving the MODEL slot empty ships no VLM at all and scores BELOW"
Write-Host "v5.2. And do not grab surgvu26-models.tar.gz -- that is v5.2's own"
Write-Host "tarball, five characters different and within 2 MB of the same size."
