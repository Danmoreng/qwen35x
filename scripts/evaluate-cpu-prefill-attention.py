"""Correctness/quality runs only; do not use their timings as benchmarks.

Frozen Q4 vs candidate, and both against a llama.cpp BF16 CPU teacher. Each invocation
does a real batched prefill followed by a fixed, teacher-forced continuation.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',default='build-prefill-tiled/qwen35x.exe')
    p.add_argument('--baseline',required=True)
    p.add_argument('--teacher',required=True)
    p.add_argument('--out',required=True,type=pathlib.Path)
    p.add_argument('--chunk',default=0,type=int)
    p.add_argument('--query-tile',default=0,type=int)
    p.add_argument('--kv-tile',default=32,type=int)
    p.add_argument('--mode',choices=['auto','tiled'],default='auto')
    p.add_argument('--gqa',action='store_true')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    metadata={key:str(value) for key,value in vars(args).items()}
    metadata['sha256']={str(path):digest(path) for path in [args.baseline,args.candidate,args.teacher,
        'models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h',
        'models/gguf/qwen3.5-0.8b-review-target-bf16.gguf']}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/qwen3.5-0.8b',local_files_only=True)
    cases=[
        ('narrative', '\n'.join(f'On day {i}, Mira checked the river gauge, repaired a wooden bridge and recorded the weather. The water was clear and the northern road remained open.' for i in range(45))+'\nSummarize the work in one sentence.\n',
         'Mira monitored the river, repaired the bridge, and recorded the weather while the northern road stayed open.'),
        ('ids', 'Read these archive entries carefully.\n'+'\n'.join(f'Entry {i}: owner Team-{i%17}, archive ID AX-{71003+i*13}, quantity {i%31+1}, status verified.' for i in range(145))+'\nWhat is the archive ID and quantity for entry 73? Answer with the values only.\n',
         'AX-71952, quantity 12. The record is marked verified and its owner is Team-5.'),
        ('structured_recall', 'The emergency access phrase is silver meadow. Preserve it exactly.\n'+'\n'.join(json.dumps({'record':i,'zone':['north','south','east','west'][i%4],'enabled':i%3!=0,'value':i*7%101}) for i in range(160))+'\nReturn JSON with the access phrase and the total number of records.\n',
         '{"access_phrase":"silver meadow","records":160,"note":"The phrase was given before the list of records."}'),
    ]
    summary={}
    for name,prompt,continuation in cases:
        prompt_path=args.out/f'{name}.txt';prompt_path.write_text(prompt,encoding='utf-8')
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
        forced=tokenizer.encode(continuation,add_special_tokens=False)
        token_path=args.out/f'{name}.tokens.csv'
        token_path.write_text(','.join(map(str,prompt_ids)),encoding='utf-8')
        for variant,exe in [('baseline',args.baseline),('candidate',args.candidate),('bf16',args.teacher)]:
            prefix=args.out/f'{name}-{variant}'
            cmd=[exe,'--hf-model-dir','models/qwen3.5-0.8b','--prompt-file',str(prompt_path),
                 '--forced-output-text',continuation,'--max-context','12288',
                 '--logits-out',str(prefix)+'.bin','--profile-json',str(prefix)+'.json']
            if variant=='bf16':
                cmd=[exe,'--cpu-gguf','models/gguf/qwen3.5-0.8b-review-target-bf16.gguf',
                     '--prompt-tokens-file',str(token_path),'--forced-output-tokens',','.join(map(str,forced)),
                     '--max-new-tokens',str(len(forced)),'--max-context','12288','--cpu-threads','8',
                     '--logits-out',str(prefix)+'.bin','--profile-json',str(prefix)+'.json']
            else:
                cmd+=['--infer-reference','--cpu-q4-h128','models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h','--cpu-threads','8']
                if variant=='candidate':
                    cmd+=['--cpu-attention',args.mode,'--cpu-prefill-chunk-size',str(args.chunk),
                          '--cpu-attention-query-tile',str(args.query_tile),'--cpu-attention-kv-tile',str(args.kv_tile)]
                    if args.gqa:cmd+=['--cpu-attention-gqa']
            with (args.out/f'{name}-{variant}.log').open('w',encoding='utf-8') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'Completed quality {name}/{variant}',flush=True)
            profile=json.loads(pathlib.Path(str(prefix)+'.json').read_text(encoding='utf-8'))
            if profile['prompt_tokens']!=len(prompt_ids) or profile['generated_tokens']!=len(forced):
                raise ValueError('Tokenizer/count mismatch')
        for teacher,candidate in [('baseline','candidate'),('bf16','baseline'),('bf16','candidate')]:
            out=args.out/f'{name}-{teacher}-vs-{candidate}.json'
            subprocess.run([sys.executable,'scripts/compare-logit-dumps.py',
                '--teacher',str(args.out/f'{name}-{teacher}.bin'),
                '--candidate',str(args.out/f'{name}-{candidate}.bin'),'--json',str(out)],check=True,stdout=subprocess.DEVNULL)
            summary[f'{name}/{teacher}-vs-{candidate}']=json.loads(out.read_text(encoding='utf-8'))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
