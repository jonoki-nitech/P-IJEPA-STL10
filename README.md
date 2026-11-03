# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_sigmasample.log
```

## 成果物の保存先

### サンプリングlookahead実験(`sigma_lookahead_sample`)の成果物

| 成果物                          | 保存先                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 選択シーケンス                  | `selected_patch_index/sigmasample_k{K}/seed_{s}/block_{b}_tmax_32_sigma_lookahead_sample.json`        |
| 再構成誤差                      | `images/rec_images_sigmasample_k{K}/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`                       |
| 不確実性トラジェクトリ          | `images/rec_images_sigmasample_k{K}/seed_{s}/b{b}/sigma_sum_traj.pt`(σ和・σrank)                     |
| 旧σ実験の初期ブロック(復元)     | `init_blocks/seed_{s}.pt`(zipに同梱)                                                                 |
| まとめzip(実行終了時に自動生成) | `result_sigmasample_k{K}_seed_{s}.zip` ×{各seed} + K別の`result_sigmasample_k{K}.zip` + 全部まとめた**`result.zip`**(共有はこれ一つでOK) |

既定は **K=1 と K=4** の2構成を別runで実行(K=1=候補ごとにεを1回引く素の
1サンプルlookahead、K=4=4サンプル平均を規準にしたlookahead)。成果物はKで
ディレクトリが分かれるので上書きはしません。σ_greedy/σ_lookaheadの曲線は旧σ実験
(`results/STL10_1`)の成果物をそのまま使うので、zipには新variantの成果物のみ入ります
(calibrationも旧σ実験のまま流用)。
