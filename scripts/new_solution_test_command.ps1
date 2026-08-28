python -m main.main_filename_based_solution --hf-dataset Linq-AI-Research/FinDER --data-dir data\10k\converted --default-year 2023 --column text --id-column _id --output-csv --concurrent 10 --start-row 1 --end-row 100




# On Perry's working computer
python -m main.main_filename_based_solution --csv c:\Users\dev\RAG-pipeline\random_queries_csv\Linq-AI-Research_FinDER-300-01.csv --data-dir c:\Users\dev\RAG-pipeline\data\10k\converted --default-year 2023 --column query --id-column query_id --output-csv --concurrent 10