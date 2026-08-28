python -m main.main_benchmark_rag_result `
    --input-csv c:\Users\dev\RAG-pipeline\output\Linq-AI-Research_FinDER-300-01-1-to-300-gpt-4.1-20260121_160659.csv `
    --hf-dataset Linq-AI-Research/FinDER `
    --gt-field answer `
    --id-field _id `
    --output-csv `
    --concurrent 10