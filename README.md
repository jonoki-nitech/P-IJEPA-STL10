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

### サンプリングlookahead実験(`sigma_lookahead_sample*`)の成果物

| 成果物                          | 保存先                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 選択シーケンス                  | `selected_patch_index/sigmasample_msn_k{K}/seed_{s}/block_{b}_tmax_32_{variant}.json`                |
| 再構成誤差                      | `images/rec_images_sigmasample_msn_k{K}/seed_{s}/b{b}/rec_error_vs_t.{png,csv,pt}`                   |
| 不確実性トラジェクトリ          | `images/rec_images_sigmasample_msn_k{K}/seed_{s}/b{b}/sigma_sum_traj.pt`(σ和・σrank)                 |
| 旧σ実験の初期ブロック(復元)     | `init_blocks/seed_{s}.pt`(zipに同梱)                                                                 |
| まとめzip(実行終了時に自動生成) | `result_sigmasample_msn_k{K}_seed_{s}.zip` ×{各seed} + `result_sigmasample_msn_k{K}.zip` + 全部まとめた**`result.zip`**(共有はこれ一つでOK) |

既定の実行: variant **`sigma_lookahead_sample_msn`**(ネスト版・深さ2), **K=1**。
ロールアウト内の次block選択も1-step候補と同じ方法: state1の完全未観測アンカー上位M個を
**それぞれ模擬追加**(追加特徴はその状態分布からサンプル)してσ和を評価し、**最小のアンカー**
が模擬上の次手として選ばれる。スコア(c1規準)=「c1後の続き1手を最善に引いた世界の
最終σ和」のK本平均。フォワード数 ~M(2+M)で 選択は旧σ-lookaheadの15–25倍
(seedあたり8–14h見込み)。`KS`/`LOOKAHEAD_M`(=c2の候補数も兼ねる)で任意変更。
greedy続き版`sigma_lookahead_sample_ms`(--lookahead-steps L=2, コスト~3x)と
1-step版`sigma_lookahead_sample`(L=1, データあり:
`results/STL10_lookahead_sample`)は別variantで残してあり、成果物は
`sigmasample_msn_k{K}` / `sigmasample_L{L}_k{K}` ディレクトリに分離される。
σ_greedy/σ_lookaheadの曲線は旧σ実験(`results/STL10_1`)の成果物をそのまま使い、
zipには新variantの成果物のみ入ります(calibrationも旧σ実験のまま流用)。
