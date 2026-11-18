$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildEnvironment = Join-Path $projectRoot ".venv-build"
$python = Join-Path $buildEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    & py -3.12 -m venv $buildEnvironment
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $python)) {
        throw "Python 3.12 est requis pour produire l'exécutable Windows."
    }
}

Push-Location $projectRoot
try {
    $version = (& $python -c "import sys; sys.path.insert(0, 'src'); import dessinopia; print(dessinopia.__version__)").Trim()
    if (-not $version) {
        throw "La version de Dessinopia n'a pas pu être déterminée."
    }
    if ($version -match '^(\d+)\.(\d+)\.(\d+)b(\d+)$') {
        $windowsVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3]).$($Matches[4])"
        $releaseVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3])-beta.$($Matches[4])"
    }
    else {
        $windowsVersion = "$version.0"
        $releaseVersion = $version
    }
    & $python -m pip install -e ".[build]"
    if ($LASTEXITCODE -ne 0) {
        throw "L'installation des dépendances de build a échoué (code $LASTEXITCODE)."
    }
    & $python -m nuitka `
        --mode=standalone `
        --mingw64 `
        --assume-yes-for-downloads `
        --enable-plugin=pyside6 `
        --include-qt-plugins=multimedia `
        --windows-console-mode=disable `
        --windows-icon-from-ico=src/dessinopia/assets/dessinopia.ico `
        --include-data-files=src/dessinopia/assets/palette.png=dessinopia/assets/palette.png `
        --company-name=Slyreus `
        --product-name=Dessinopia `
        --file-version=$windowsVersion `
        --product-version=$windowsVersion `
        --file-description="Dessinopia - assistant de peinture Heartopia" `
        --copyright="Copyright (c) 2026 Slyreus" `
        --output-dir=dist `
        --output-filename=Dessinopia.exe `
        src\dessinopia\app.py
    if ($LASTEXITCODE -ne 0) {
        throw "La compilation Nuitka a échoué (code $LASTEXITCODE)."
    }

    $standaloneDir = Join-Path $projectRoot "dist\app.dist"
    Copy-Item -LiteralPath (Join-Path $projectRoot "README.md") -Destination $standaloneDir `
        -Force
    Copy-Item -LiteralPath (Join-Path $projectRoot "LICENSE") -Destination $standaloneDir `
        -Force
    $archiveName = "Dessinopia-$releaseVersion-win64.zip"
    $archive = Join-Path $projectRoot "dist\$archiveName"
    Compress-Archive -Path (Join-Path $standaloneDir "*") -DestinationPath $archive `
        -CompressionLevel Optimal -Force
    $checksumPath = Join-Path $projectRoot "dist\Dessinopia-$releaseVersion-win64.sha256"
    $checksum = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    "$checksum  $archiveName" | Set-Content -LiteralPath $checksumPath `
        -Encoding ascii

    $defenderPlatform = Join-Path $env:ProgramData "Microsoft\Windows Defender\Platform"
    $defender = Get-ChildItem -LiteralPath $defenderPlatform -Recurse `
        -Filter "MpCmdRun.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\X86\\" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($defender) {
        foreach ($file in @((Join-Path $standaloneDir "Dessinopia.exe"), $archive)) {
            & $defender.FullName -Scan -ScanType 3 -File $file -DisableRemediation
            if ($LASTEXITCODE -ne 0) {
                throw "Microsoft Defender a refusé le fichier : $file"
            }
        }
    }
    else {
        Write-Warning "Microsoft Defender est introuvable : analyse antivirus non effectuée."
    }
    Write-Host "Archive créée : $archive"
}
finally {
    Pop-Location
}
