# 実行手順

## 事前準備

- 学習済みモデルをHugging Faceからダウンロードし、以下の場所に置いていただきたいです。
  - https://huggingface.co/jonoki-nitech/probabilistic_I-JEPA_STL10 ("Files and Versions" → `training_model.pth`, `latest.ckpt`)
  - `training_model.pth` → `models/embdim=512_epoch=300_patchsize=8/training_model.pth`
  - `latest.ckpt` → `models/mae/latest.ckpt`

- 仮想環境: `p-ijepa_stl10`

## 実行

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_sigmasample_enum2.log
```

## 成果物の保存先

### サンプリングlookahead実験(`sigma_lookahead_sample*`)の成果物

| 成果物                          | 保存先                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 選択シーケンス                  | `selected_patch_index/sigmasample_enum2_k{K}/seed_{s}/block_{b}_tmax_32_{variant}.json`               |
| 再構成誤差                      | `images/rec_images_sigmasample_enum2_k{K}/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`                 |
| 不確実性トラジェクトリ          | `images/rec_images_sigmasample_enum2_k{K}/seed_{s}/b{b}/sigma_sum_traj.pt`(σ和・σrank)               |
| 旧σ実験の初期ブロック(復元)     | `init_blocks/seed_{s}.pt`(zipに同梱)                                                                 |
| まとめzip(実行終了時に自動生成) | `result_sigmasample_enum2_k{K}_seed_{s}.zip` ×{各seed} + `result_sigmasample_enum2_k{K}.zip` + 全部まとめた**`result.zip`**(共有はこれ一つでOK) |

既定の実行: variant **`sigma_lookahead_sample_enum2`**(列挙型・深さ2), **K=1**。
論文(frontiers.tex)の **expected uncertainty minimization**(H=2)の実装:
1. 現在のσ mapの上位M個をc1候補に
2. 各c1を**平均μで模擬追加**(deterministic)し、そのσ mapの上位M個をc2候補に
   → **M×M=256の候補系列**
3. 各系列を**独立のK=1サンプル鎖**でシミュレーション(2つの追加がどちらも
   予測分布からのサンプル, chain rule)し、**時点Tの未観測パッチのみのΣσ²**
   を期待不確実性として計算
4. スコア最小の系列(=期待不確実性最小)の**2ブロックをそのまま観測**(ms2と同じ
   ペア観測; 最終1ブロックのみの場合はc1単体)。t軸は「観測済みblock数」のまま。
c2候補のパネルは`--M`(既定16)と共有。
forward ~(1+2M)=33×/real stepだがreal stepが半分 → 選択コストは
旧σ-lookaheadの約16倍(**選択~8–14h/seed**)。`KS`/`LOOKAHEAD_M`(=c2パネル幅)で
任意変更。`CHUNK`既定は96GB GPU想定の8192(24GBカードなら1024–4096に切戻し)。
成果物は`sigmasample_enum2_k{K}`ディレクトリに分離
(ms2=`sigmasample_ms2_k{K}`, msn=`sigmasample_msn_k{K}`,
greedy ms=`sigmasample_L{L}_k{K}`, 1-step版データは
`results/STL10_lookahead_sample`にあり)。σ_greedy/σ_lookaheadの曲線は旧σ実験
(`results/STL10_1`)の成果物をそのまま使い、zipには新variantの成果物のみ入ります
(calibrationも旧σ実験のまま流用)。
