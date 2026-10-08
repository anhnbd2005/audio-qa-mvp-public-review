# ViMedCSS V3 PlanOnly runner (HUMAN-OPERATED, SAFE BY CONSTRUCTION).
#
# Produces the deterministic semantic PLAN for one authorized split and audits
# it. It NEVER renders QA and NEVER freezes a release. Full production is a
# separate explicit human command (see the V3 report).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\autonomous_qa\run_vimedcss_v3_plan.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\autonomous_qa\run_vimedcss_v3_plan.ps1 -Split train

param(
    [string]$Split = "train"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
$env:PYTHONPATH = $Root.Path

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config = "configs/qa_generation_vimedcss_v3.yaml"
$Scratch = "outputs/_scratch/vimedcss_v3/$Split"
$RunId = "plan"
$RunDir = Join-Path $Scratch $RunId

Write-Host "[1/2] plan-only ($Split)"
& $Python -m src.autonomous_qa.production.generate_qa --dataset vimedcss --config $Config --split $Split --plan-only --output-root $Scratch --run-id $RunId
if ($LASTEXITCODE -ne 0) { throw "PLAN_FAILED ($Split)" }

Write-Host "[2/2] audit final plan ($Split)"
& $Python "scripts/autonomous_qa/audit_vimedcss_v3_final.py" --run-dir $RunDir --config $Config --split $Split
if ($LASTEXITCODE -ne 0) { throw "PLAN_AUDIT_FAILED ($Split)" }

Write-Host "PLANONLY OK ($Split). No QA rendered; no release frozen."
