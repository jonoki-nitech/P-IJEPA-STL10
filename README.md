# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

### MC-Dropoutベースラインの学習(`revision/run.sh`の現行スクリプト)

```bash
conda activate p-ijepa_stl10
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee train_mcdropout.log
```

## 成果物の保存先

### MC-Dropout学習(`revision/run.sh`)

| 成果物                             | 保存先                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------- |
| 学習済みI-JEPA(σキー込み・σ未学習) | `models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth` |
| target encoder(EMA)                | `models/mcdropout_embdim=512_epoch=300_patchsize=8/encoder.pth`        |
| 上記をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                                      |
