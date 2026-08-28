import csv, glob
for g in ['01','02','03','04','05']:
    f = sorted(glob.glob(f'output/benchmark-Linq-AI-Research_FinDER-cbrllm-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv'))[-1]
    n_total=0; n_fail=0; n_perfect=0; n_correct7=0; n_correct8=0
    with open(f, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            s = (row.get('score') or '').strip()
            try: si=int(s)
            except: continue
            if 1<=si<=10:
                n_total+=1
                if si==1: n_fail+=1
                if si==10: n_perfect+=1
                if si>=7: n_correct7+=1
                if si>=8: n_correct8+=1
    print(f'group {g}: n={n_total} fail={n_fail}({n_fail/n_total*100:.2f}%) c7={n_correct7}({n_correct7/n_total*100:.2f}%) c8={n_correct8}({n_correct8/n_total*100:.2f}%) perfect={n_perfect}({n_perfect/n_total*100:.2f}%)')
