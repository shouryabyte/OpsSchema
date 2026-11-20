"""Actual CUDA PEFT entry point; it never labels CPU smoke validation as training."""
import argparse,inspect,json
from pathlib import Path
def prompt(x): return 'Extract exactly JSON keys category, priority, action, system, summary.\nRequest: '+x+'\nJSON:'
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',default='configs/train.json');a.add_argument('--cpu-smoke',action='store_true');z=a.parse_args();c=json.loads(Path(z.config).read_text());out=Path('results');out.mkdir(exist_ok=True)
 if z.cpu_smoke:
  from transformers import AutoTokenizer
  (out/'run_environment.json').write_text(json.dumps({'status':'CPU_PIPELINE_VALIDATION_ONLY','tokens':len(AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')(prompt('VPN outage'))['input_ids']),'config':c},indent=2));return
 import torch
 if not torch.cuda.is_available():raise SystemExit('CUDA unavailable: use Colab; --cpu-smoke is validation only.')
 from datasets import load_dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer,DataCollatorForLanguageModeling,BitsAndBytesConfig
 from peft import LoraConfig,get_peft_model
 t=AutoTokenizer.from_pretrained(c['model_id']);t.pad_token=t.eos_token
 m=AutoModelForCausalLM.from_pretrained(c['model_id'],device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16));m.gradient_checkpointing_enable();m.config.use_cache=False;m=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,target_modules=c['target_modules'],task_type='CAUSAL_LM'))
 def enc(r):return t(prompt(r['request'])+json.dumps(r['target'])+t.eos_token,truncation=True,max_length=c['max_length'])
 d=load_dataset('json',data_files={'train':'dataset/train.jsonl','validation':'dataset/validation.jsonl'}).map(enc,remove_columns=['id','request','target'])
 args=dict(output_dir=str(out/'checkpoints'),num_train_epochs=c['epochs'],learning_rate=c['learning_rate'],per_device_train_batch_size=1,gradient_accumulation_steps=c['gradient_accumulation_steps'],save_strategy='epoch',logging_steps=10,report_to=[],bf16=True,seed=c['seed'])
 args['eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments).parameters else 'evaluation_strategy']='epoch'
 q=TrainingArguments(**args)
 tr=Trainer(model=m,args=q,train_dataset=d['train'],eval_dataset=d['validation'],data_collator=DataCollatorForLanguageModeling(t,mlm=False));tr.train();tr.save_model(str(out/'adapter'));t.save_pretrained(str(out/'adapter'));tr.save_state();(out/'actual_train_config.json').write_text(json.dumps(c,indent=2))
if __name__=='__main__':main()
