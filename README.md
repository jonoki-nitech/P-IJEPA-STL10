# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

### MC-Dropoutモデルの評価(不確実性=MC分散)

MC評価パイプラインは `revision/archive/runmcdropout_infer.sh`(使用法はスクリプト先頭):

```bash
DEVICE=cuda:1 bash revision/archive/runmcdropout_infer.sh 2>&1 | tee eval_mcdropout.log
```

### 追加実験: サンプリングlookahead(σ-headモデル)

`sigma_lookahead_sample`: `sigma_lookahead`と同一の機構(σプリスクリーン, 候補M, dup-free
truncation, σトラジェクトリ記録)だけが、1步先の局面を評価する際の候補の模擬特徴が
**予測平均μではなく予測分布からのサンプル** `z ~ N(μ, σ̂²)`。各候補を K=16 の
サンプルでK回シミュレーションし(乱数は`--mc-seed`で再現性確保)、未観測σ²和の
**平均値**を規準に選ぶ。比較は旧データを流用(--points_ptでオーバレイ)。σ-head学習済みチェックポイント専用
(`models/mcdropout_...`パス名で検出し起動時に拒否; MC ckptのσ headは未学習なので
サンプリングが退化する)。center初期はなし。

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_sigmasample.log
```

既定: σモデル(`models/embdim=512_epoch=300_patchsize=8/training_model.pth`)で
variants=`sigma_lookahead_sample`のみを新規実行(`sigma_greedy`/`sigma_lookahead`は
旧実験の成果物をそのまま流用するので、run内には入れない)。
`N=0`/`BS="2 3 4"`/`SEEDS="0 1 2 3 4"`/`K=16`。初期ブロックは旧σ実験から復元
(`OLD_JSON_BASE`が既定とは違う場所なら指定、SKIP_INIT=1で揃えない選択も可)。
成果物はmcdropout評価と同形式で `selected_patch_index/sigmasample/…`,
`images/rec_images_sigmasample/…` に出ます(まとめzipは`result_sigmasample.zip`)。

旧σ実験との比較オーバレイの例:

```bash
python patch_selection_reconstruction_eval.py \
    --points_pt results/STL10_1/result_seed_0/images/rec_images_tmax32/seed_0/b2/rec_error_vs_t.pt \
                result_sigmasample_seed_0.zip を展開した先/images/rec_images_sigmasample/seed_0/b2/rec_error_vs_t.pt \
    --out_dir images/compare_sigmasample
```
(注意: 各runのaligned_max_tはtruncationの違いでずれ得るので、比較は共通min t以内で)

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

### サンプリングlookahead実験(`sigma_lookahead_sample`)の成果物

| 成果物                 | 保存先                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| 選択シーケンス         | `selected_patch_index/sigmasample/seed_{s}/block_{b}_tmax_32_sigma_lookahead_sample.json` |
| 再構成誤差             | `images/rec_images_sigmasample/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`            |
| 不確実性トラジェクトリ | `images/rec_images_sigmasample/seed_{s}/b{b}/sigma_sum_traj.pt`(σ和・σrank)         |
| 旧σ実験の初期ブロック(復元) | `init_blocks/seed_{s}.pt`(zipに同梱)                                            |
| まとめzip(実行終了時に自動生成) | `result_sigmasample_seed_{s}.zip` ×5 + ネスト収納した`result_sigmasample.zip`   |

備考: この実験はcalibrationを新規作成しない(σモデルは旧実験と同一チェックポイントなので
旧σ実験の`images/calibration/`成果物を流用)。σ_greedy/σ_lookaheadの曲線も旧σ実験
(`results/STL10_1`)の成果物をそのまま使うので、zipには新variantの成果物のみ入ります。
