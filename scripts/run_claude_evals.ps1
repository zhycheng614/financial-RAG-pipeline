# Phase 5 — run Claude Sonnet 4.6 eval on all 6 systems' 100-query subsets.
# Direct python.exe (conda run mangles multi-line args); UTF-8 stdout forced for Windows consoles.
# Note: do NOT use `2>&1` here — on PowerShell 5.1 redirecting a native exe's
# stderr wraps each line as NativeCommandError and trips ErrorActionPreference.

$env:PYTHONIOENCODING = "utf-8"

$repo = "."
$python = "python"   # or an absolute path to your interpreter
$systems = @("cbr", "sfr", "hdrr", "cbrmeta", "cbrllm", "agentic")

Set-Location "$repo\src"

foreach ($sys in $systems) {
    $in  = "..\output\cross_eval_input_$sys.csv"
    $out = "..\output\claude_eval_$sys.csv"
    Write-Host "===> $sys start $(Get-Date -Format 'HH:mm:ss')"
    & $python main\main_benchmark_rag_result.py `
        --input-csv $in `
        --hf-dataset Linq-AI-Research/FinDER `
        --gt-field answer `
        --id-field _id `
        --model claude-sonnet-4-6 `
        --concurrent 3 `
        --output-csv $out
    if ($LASTEXITCODE -ne 0) {
        Write-Host "===> $sys FAILED with exit $LASTEXITCODE"
    } else {
        Write-Host "===> $sys done  $(Get-Date -Format 'HH:mm:ss')"
    }
}

Write-Host "===> ALL_SYSTEMS_COMPLETE"
