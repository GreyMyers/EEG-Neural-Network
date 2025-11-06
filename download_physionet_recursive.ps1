# PowerShell script to recursively download PhysioNet EEG dataset
# This mimics wget -r behavior

$baseUrl = "https://physionet.org/files/eegmmidb/1.0.0/"
$outputDir = "C:\Users\greym\Xavier\physionet_data\eegmmidb\1.0.0"

# Create output directory
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "Downloading PhysioNet dataset recursively..."
Write-Host "This will download all files from the dataset (~1.9 GB)"
Write-Host ""

# Download RECORDS file first to get list of all files
$recordsUrl = $baseUrl + "RECORDS"
$recordsPath = Join-Path $outputDir "RECORDS"

Write-Host "Downloading RECORDS file..."
try {
    Invoke-WebRequest -Uri $recordsUrl -OutFile $recordsPath -UseBasicParsing
    Write-Host "[OK] Downloaded RECORDS" -ForegroundColor Green
} catch {
    Write-Host "Error downloading RECORDS: $_" -ForegroundColor Red
    exit 1
}

# Read RECORDS to get list of all files
$records = Get-Content $recordsPath | Where-Object { $_ -notmatch '^#' -and $_.Trim() -ne '' }
$totalFiles = $records.Count
$currentFile = 0
$failedFiles = @()

Write-Host "Found $totalFiles files to download"
Write-Host ""

foreach ($record in $records) {
    $currentFile++
    $record = $record.Trim()
    if ([string]::IsNullOrWhiteSpace($record)) { continue }
    
    # Create directory structure if needed
    $relativePath = $record
    $filePath = Join-Path $outputDir $relativePath
    $fileDir = Split-Path $filePath -Parent
    
    if (-not (Test-Path $fileDir)) {
        New-Item -ItemType Directory -Force -Path $fileDir | Out-Null
    }
    
    # Skip if file already exists
    if (Test-Path $filePath) {
        Write-Host "[$currentFile/$totalFiles] Skipping $record (already exists)" -ForegroundColor Yellow
        continue
    }
    
    # Download .edf file
    $edfUrl = $baseUrl + $record
    Write-Host "[$currentFile/$totalFiles] Downloading $record..."
    
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $edfUrl -OutFile $filePath -UseBasicParsing -TimeoutSec 300
        Write-Host "  [OK] Downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  [ERROR] Error: $_" -ForegroundColor Red
        $failedFiles += $record
        continue
    }
    
    # Also try to download .event file (annotations)
    $eventPath = $filePath -replace '\.edf$', '.event'
    $eventUrl = $edfUrl -replace '\.edf$', '.event'
    
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $eventUrl -OutFile $eventPath -UseBasicParsing -TimeoutSec 60 -ErrorAction SilentlyContinue
    } catch {
        # .event files are optional, so we don't report errors
    }
    
    # Progress update every 10 files
    if ($currentFile % 10 -eq 0) {
        $percent = [math]::Round(($currentFile / $totalFiles) * 100, 1)
        Write-Host "  Progress: $percent% ($currentFile/$totalFiles files)" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "=== Download Complete ===" -ForegroundColor Green
Write-Host "Downloaded: $($totalFiles - $failedFiles.Count) files"
if ($failedFiles.Count -gt 0) {
    Write-Host "Failed: $($failedFiles.Count) files" -ForegroundColor Red
    Write-Host "Failed files saved to: failed_downloads.txt"
    $failedFiles | Out-File "failed_downloads.txt"
}

Write-Host ""
Write-Host "Files saved to: $outputDir"
Write-Host ""
Write-Host "You can now run the extraction script:"
Write-Host "  python src\extract_physionet.py --physionet-dir `"$outputDir`" --user-name physionet"

