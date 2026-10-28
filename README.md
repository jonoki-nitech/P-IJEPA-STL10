# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

### MC-Dropoutモデルの評価(不確実性=MC分散)

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_mcdropout.log
```

## 成果物の保存先

### MC-Dropout評価の成果物

| 成果物                         | 保存先                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------- |
| 選択シーケンス                 | `selected_patch_index/mcdropout/seed_{s}/block_{b}_tmax_32_{mc_*}.json`       |
| 再構成誤差                     | `images/rec_images_mcdropout/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`      |
| 不確実性トラジェクトリ         | `images/rec_images_mcdropout/seed_{s}/b{b}/sigma_sum_traj.pt`(MC分散和)      |
| calibration                    | `images/calibration_mcdropout/calibration_points.pt`(ほか既存形式と同一)      |
| 旧σモデルとの比較オーバレイ    | `images/compare_sigma_vs_mcdropout/rec_error_vs_t.compare.{png,pt}`           |
| 旧実験から復元した初期ブロック | `init_blocks/seed_{s}.pt`(`--init-blocks-pt`で入力、成果物というより共有素材) |
| まとめzip(実行終了時に自動生成) | `result_seed_{s}.zip` ×5 + ネスト収納した`result.zip`(学習時の`result.zip`を上書き) |
