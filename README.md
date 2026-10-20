# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

```bash
conda activate p-ijepa_stl10
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee run.log
```

(CHUNKでチャンクサイズを変更可能です．デフォルトは1024ですが，メモリが足りない場合はお手数ですが減らして実行をお願いいたします．)

## 成果物の保存先

| 成果物                                                                  | 保存先                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| σのcalibration(図表+metrics CSV+後分析用pt、seed非依存・共通)           | `images/calibration/`                                         |
| 9戦略の選択系列JSON(b=1 × 8変種 / b=2〜4 × 9変種)                       | `selected_patch_index/seed_{s}/block_{1..4}_tmax_32_*.json`   |
| MAE再構成誤差 vs t(グラフPNG / CSV / 生値pt、同一alignedグリッドで統一) | `images/rec_images_tmax32/seed_{s}/b{1..4}/`                  |
| 画像ごとの不確実性総和の推移 + lookaheadの選択ブロックσ順位             | `images/rec_images_tmax32/seed_{s}/b{1..4}/sigma_sum_traj.pt` |
| 再構成画像(最初の10サンプル、最終t)                                     | `images/rec_images_tmax32/seed_{s}/b{1..4}/gallery/`          |
| 上記成果物+共通calibrationをまとめたzip(各seedの実行終了時に自動生成)   | `result_seed_{s}.zip`(リポジトリ直下)                         |
| すべての`result_seed_{s}.zip`をまとめたネストzip(実行終了時に自動生成)  | `result.zip`(リポジトリ直下)                                  |
