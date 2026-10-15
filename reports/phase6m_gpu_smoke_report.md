# Phase 6M GPU smoke

{
  "schema_version": "phase6m.gpu_smoke.v1",
  "status": "GPU_SMOKE_PASS_FULL_TRAINING_READY",
  "subprocess_returncode": 0,
  "cuda_probe": [
    "True",
    "NVIDIA H800"
  ],
  "steps": 2,
  "loss": 4.668266773223877,
  "peak_memory_bytes": 836897792,
  "checkpoint_reload": true,
  "checkpoint_sha256": "1196dc80abd4a6907517aff173d5c0b75b4d35a95cd0957046ffc219ca6e65e8",
  "corpus_sha256": "fbc44e2c343570c5c6e07f5de76fa781956bb1d795b9a981f330c88774ec70e2",
  "training_commit": "18c33147695733db74c888e6e594a9f5c35601e3",
  "stderr_tail": "(out / \"vocab.json\").write_text(json.dumps(vocab, sort_keys=True), encoding=\"utf-8\"); mc = cfg[\"model\"]; opt_cfg = cfg[\"training\"]; corpus_sha = file_sha(train_path); config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest(); commit = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=ROOT, text=True).strip(); model = PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab), **mc)).to(device); head = MLMHead(mc[\"hidden_size\"], len(vocab)).to(device); opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=opt_cfg[\"lr\"], weight_decay=opt_cfg.get(\"weight_decay\", 0.01)); scaler = torch.cuda.amp.GradScaler(enabled=device.type == \"cuda\"); step = 0; start_epoch = 0\n/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/pm/pm2/scripts/pretrain_pi1m.py:70: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.\n  with torch.cuda.amp.autocast(enabled=device.type == \"cuda\"):\n"
}
