"""Summarize sequential prefill A/B runs without dropping outliers."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--out-prefix',required=True,type=Path)
    args=p.parse_args()
    metadata=json.loads((args.directory/'metadata.json').read_text(encoding='utf-8-sig'))
    samples=[]; groups={}
    for file in sorted(args.directory.glob('case*.csv')):
        match=re.match(r'case(\d+)-p(\d+)-',file.name)
        index,length=map(int,match.groups())
        variant=metadata['variants'][index%len(metadata['variants'])]
        rows=list(csv.DictReader(file.open(encoding='utf-8-sig',newline='')))
        if len(rows)!=metadata['runs']:raise ValueError(f'Incomplete {file}')
        for row in rows:
            if int(row['prompt_tokens'])!=length:raise ValueError('Prompt mismatch')
            profile=json.loads(Path(row['profile_json']).read_text(encoding='utf-8-sig'))
            outputs=int(row['generated_tokens'])
            steps=int(profile.get('forward_pass_tokens',length))-length if outputs else 0
            if outputs and steps!=outputs-1:raise ValueError('Decode-forward count mismatch')
            sample=dict(case=index,prompt_tokens=length,variant=variant,run=int(row['run_index']),
                prefill_ms=float(row['prefill_time_ms']),prefill_tps=float(row['prefill_tokens_per_second']),
                generated_tokens=outputs,decode_forward_steps=steps,decode_ms=float(row['decode_time_ms']),
                decode_tps=steps*1000/float(row['decode_time_ms']) if steps else 0,
                kernel=profile.get('cpu_attention_kernel','unknown'),
                chunk=profile.get('cpu_prefill_chunk_size_resolved',0))
            samples.append(sample);groups.setdefault((length,variant),[]).append(sample)
    expected=len(metadata['lengths'])*len(metadata['variants'])*metadata['runs']
    if len(samples)!=expected:raise ValueError(f'Expected {expected} samples, found {len(samples)}')
    summary=[]
    for (length,variant),rows in sorted(groups.items()):
        summary.append(dict(prompt_tokens=length,variant=variant,samples=len(rows),
            prefill_tps_median=statistics.median(r['prefill_tps'] for r in rows),
            prefill_tps_min=min(r['prefill_tps'] for r in rows),prefill_tps_max=max(r['prefill_tps'] for r in rows),
            decode_tps_median=statistics.median(r['decode_tps'] for r in rows)))
    args.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix,rows in [('samples',samples),('summary',summary)]:
        with Path(str(args.out_prefix)+f'-{suffix}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    Path(str(args.out_prefix)+'-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
