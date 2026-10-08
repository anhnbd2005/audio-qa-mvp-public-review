# ViMedCSS V2 FINAL end-to-end production runner (HUMAN-OPERATED).
#
# Thin wrapper over the EXISTING production entrypoints. It does not implement
# a second pipeline: it runs the current deterministic generator (which already
# binds approved templates and computes gold), audits the final plan, audits the
# QA, verifies exact counts, and freezes the LOGICAL release. Physical audio
# mapping stays deferred to map_vimedcss_audio_paths.py.
#
# Normal run:
#   powershell -ExecutionPolicy Bypass -File .\scripts\autonomous_qa\run_vimedcss_v2_full.ps1
# Diagnostic (plan only, no QA / no release freeze):
#   powershell -ExecutionPolicy Bypass -File .\scripts\autonomous_qa\run_vimedcss_v2_full.ps1 -PlanOnly

param(
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
$env:PYTHONPATH = $Root.Path

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config = "configs/qa_generation_vimedcss_v2.yaml"
$Scratch = "outputs/_scratch/vimedcss_v2_final"
$RunId = "final"
$RunDir = Join-Path $Scratch $RunId
$ReleaseDir = "outputs/releases/vimedcss/final"

function Invoke-Step([string]$label, [scriptblock]$action) {
    Write-Host "[$label]"
    & $action
    if ($LASTEXITCODE -ne 0) { throw "STEP_FAILED: $label (exit $LASTEXITCODE)" }
}

# 1. Verify/refresh the canonical ViMedCSS language preflight artifact.
Invoke-Step "1/6 language preflight" {
    & $Python -c "from src.autonomous_qa.language import language_preflight as pf; r=pf.run_preflight(mode='dataset', dataset='vimedcss', accepted_types=pf.get_dataset_accepted_types('vimedcss'), write_outputs=True); assert r['audit']['result']=='PREFLIGHT_PASS' and r['audit']['blocking_issue_count']==0, r['audit']"
}

# 2. Generate the deterministic semantic plan (and QA unless -PlanOnly).
if ($PlanOnly) {
    Invoke-Step "2/6 plan-only" {
        & $Python -m src.autonomous_qa.production.generate_qa --dataset vimedcss --config $Config --output-root $Scratch --run-id $RunId --plan-only
    }
} else {
    Invoke-Step "2/6 plan + render QA" {
        & $Python -m src.autonomous_qa.production.generate_qa --dataset vimedcss --config $Config --output-root $Scratch --run-id $RunId
    }
}

# 3. Audit the final plan (1:1 direct + 2+2 anchor neighborhood).
Invoke-Step "3/6 audit final plan" {
    & $Python "scripts/autonomous_qa/audit_vimedcss_v2_final.py" --run-dir $RunDir
}

if (-not $PlanOnly) {
    # 4. Audit QA (produced by the existing generator).
    Invoke-Step "4/6 audit QA" {
        & $Python -c "import json,sys; v=json.load(open(r'$RunDir/qa_validation.json',encoding='utf-8')); sys.exit(0 if v.get('all_passed') else 1)"
    }

    # 5. Verify exact final counts.
    Invoke-Step "5/6 verify exact counts" {
        & $Python -c "import json,collections,sys; rows=[json.loads(l) for l in open(r'$RunDir/qa_model_facing.jsonl',encoding='utf-8') if l.strip()]; c=collections.Counter(r['type_id'] for r in rows); exp={'vimedcss_topic_classification':11832,'vimedcss_cs_terms_count':11832,'vimedcss_pairwise_topic_same':47328}; sys.exit(0 if dict(c)==exp and len(rows)==70992 else 1)"
    }

    # 6. Freeze the LOGICAL release (count-enforced; refuses anything but final).
    Invoke-Step "6/6 freeze logical release" {
        & $Python "scripts/autonomous_qa/build_vimedcss_release.py" --run-dir $RunDir --output-dir $ReleaseDir --expected-final
    }

    Write-Host "RELEASE FROZEN: $ReleaseDir"
    Write-Host "Physical audio mapping remains DEFERRED (map_vimedcss_audio_paths.py)."
} else {
    Write-Host "PLAN-ONLY complete. QA generation and release freeze were intentionally skipped."
}
