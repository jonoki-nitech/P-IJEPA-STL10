# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

- データ(STL10)は `data_utils.py` が自動解決します。
  - 既存データ(`"/home/jonoki/research/master/byol_trial/ExtreMA_stl10/data"`)があればそれを使用(ダウンロードなし)
  - 無ければ `data/stl10_binary/` に必要な split(TEST)のみ自動で取得します

## 実行

```bash
conda activate p-ijepa_stl10
DEVICE=cuda:0 bash revision/run.sh 2>&1 | tee run.log
```

## 成果物の保存先

| 成果物                                        | 保存先                                             |
| --------------------------------------------- | -------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt)   | `images/calibration/`                              |
| 7戦略の選択系列JSON(b=1〜4 × 7変種)           | `selected_patch_index/block_{1..4}_tmax_32_*.json` |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt)  | `images/rec_images_tmax32/b{1..4}/`                |
| 再構成画像(最初の10サンプル、最終t)           | `images/rec_images_tmax32/b{1..4}/gallery/`        |
| 上記成果物をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                       |
