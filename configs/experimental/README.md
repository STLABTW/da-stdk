# Experimental / ablation configs

Configs here are not used by the main paper pipeline. Scripts can still use them by path, e.g.:

```bash
python scripts/train_default.py --config configs/experimental/config_conformal_demo.yaml
```

- **config_conformal_ratio_0.2.yaml** … **config_conformal_ratio_0.35.yaml**: Conformal calibration ratio sweep (ablations).
- **config_conformal_demo.yaml**: Short demo run (different train_ratio/epochs/lr from KAUST experiment).
