# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_sigmasample_ms2.log
```

## 成果物の保存先

### サンプリングlookahead実験(`sigma_lookahead_sample*`)の成果物

| 成果物                          | 保存先                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 選択シーケンス                  | `selected_patch_index/sigmasample_ms2_k{K}/seed_{s}/block_{b}_tmax_32_{variant}.json`                |
| 再構成誤差                      | `images/rec_images_sigmasample_ms2_k{K}/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`                   |
| 不確実性トラジェクトリ          | `images/rec_images_sigmasample_ms2_k{K}/seed_{s}/b{b}/sigma_sum_traj.pt`(σ和・σrank)                 |
| 旧σ実験の初期ブロック(復元)     | `init_blocks/seed_{s}.pt`(zipに同梱)                                                                 |
| まとめzip(実行終了時に自動生成) | `result_sigmasample_ms2_k{K}_seed_{s}.zip` ×{各seed} + `result_sigmasample_ms2_k{K}.zip` + 全部まとめた**`result.zip`**(共有はこれ一つでOK) |

既定の実行: variant **`sigma_lookahead_sample_ms2`**(ペア観測版・深さ2), **K=1**。
msnと同一のネスト計画(「c1+最善の続き1手」の2手プランを模擬でスコア化し、スコア最小の
c2を内定)で、**内定した2ブロック両方を実際に観測として追加**する(L=2のプランをそのまま
実行; 残り1ブロック分のbudgetしかない最終段階ではc1のみ)。t軸は「観測済みblock数」
のままなので、1ブロック/ステップの他variantと同一軸で比較可能。
c2候補のパネルは`--M`(既定16)と共有。ロールアウトはK本(既定K=1=単一サンプル)。
forward ~M(2+M)/real stepだがreal stepが半分になるため、選択コストはmsnの約半分
(**選択~4.5–7.5h/seed**)。`KS`/`LOOKAHEAD_M`で任意変更。成果物は
`sigmasample_ms2_k{K}`ディレクトリに分離(msn=`sigmasample_msn_k{K}`,
greedy ms=`sigmasample_L{L}_k{K}`, 1-step版データは
`results/STL10_lookahead_sample`にあり)。σ_greedy/σ_lookaheadの曲線は旧σ実験
(`results/STL10_1`)の成果物をそのまま使い、zipには新variantの成果物のみ入ります
(calibrationも旧σ実験のまま流用)。
