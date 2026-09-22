$ErrorActionPreference = 'Stop'
$auditRoot = Split-Path -Parent $PSScriptRoot

Write-Host ''
Write-Host 'Expected empty source diffs:'
git diff --no-index --name-status -- `
  (Join-Path $auditRoot 'original_author/graphcov') `
  (Join-Path $auditRoot 'current_project/table1_reproduction/vendor/graphcov')
if ($LASTEXITCODE -eq 1) {
  throw 'Author source and vendored source differ. Inspect VENDOR_VS_AUTHOR.patch.'
}

git diff --no-index --name-status -- `
  (Join-Path $auditRoot 'original_author/graphcov') `
  (Join-Path $auditRoot 'current_project/graphcov')
if ($LASTEXITCODE -eq 1) {
  throw 'Shared graphcov differs from author source. Inspect SHARED_GRAPHCOV_VS_AUTHOR.patch.'
}

Write-Host 'Both graphcov source comparisons are identical.'
