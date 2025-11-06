# PowerShell script to download PhysioNet EEG Motor Movement/Imagery Dataset
# This uses Invoke-WebRequest which is built into PowerShell

$baseUrl = "https://physionet.org/files/eegmmidb/1.0.0/"
$outputDir = "C:\Users\greym\Xavier\physionet_data\eegmmidb\1.0.0"

# Create output directory
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "Downloading PhysioNet dataset..."
Write-Host "Note: This will download individual files. For faster download, consider using the ZIP file instead."
Write-Host ""

# Download RECORDS file first to get list of files
$recordsUrl = $baseUrl + "RECORDS"
$recordsPath = Join-Path $outputDir "RECORDS"
try {
    Invoke-WebRequest -Uri $recordsUrl -OutFile $recordsPath
    Write-Host "Downloaded RECORDS file"
} catch {
    Write-Host "Error downloading RECORDS: $_"
    exit 1
}

# Read RECORDS to get list of subject files
$records = Get-Content $recordsPath
$totalFiles = $records.Count
$currentFile = 0

foreach ($record in $records) {
    $currentFile++
    $record = $record.Trim()
    if ([string]::IsNullOrWhiteSpace($record)) { continue }
    
    # Skip comment lines
    if ($record.StartsWith("#")) { continue }
    
    # Create subject directory if needed
    $subjectDir = Join-Path $outputDir $record.Split('/')[0]
    New-Item -ItemType Directory -Force -Path $subjectDir | Out-Null
    
    # Download .edf file (record already contains .edf extension)
    $edfUrl = $baseUrl + $record
    $edfPath = Join-Path $outputDir $record
    
    Write-Host "[$currentFile/$totalFiles] Downloading $record..."
    
    try {
        Invoke-WebRequest -Uri $edfUrl -OutFile $edfPath -UseBasicParsing -TimeoutSec 300
    } catch {
        Write-Host "  Error downloading $record: $_"
        continue
    }
    
    # Download .event file (annotations)
    $eventUrl = $baseUrl + ($record -replace '\.edf$', '.event')
    $eventPath = Join-Path $outputDir ($record -replace '\.edf$', '.event')
    
    try {
        Invoke-WebRequest -Uri $eventUrl -OutFile $eventPath
    } catch {
        Write-Host "  Warning: Could not download $record.event"
    }
}

Write-Host ""
Write-Host "Download complete! Files saved to: $outputDir"
Write-Host "You can now run: python src\extract_physionet.py --physionet-dir `"$outputDir`" --user-name physionet"

