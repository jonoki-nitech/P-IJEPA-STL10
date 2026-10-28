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

### MC-Dropoutモデルの評価(不確実性=MC分散)

本番(5seed)の一括実行は`revision/run.sh`(旧σスイープと同一プロトコル・同一初期ブロック
のオーケストレーション + zipパッケージ化):

```bash
DEVICE=cuda:1 bash revision/run.sh 2>&1 | tee eval_mcdropout.log
```

既定は`N=0`(=8000)、`BS="2 3 4"`(b=1は除外)、`SEEDS="0 1 2 3 4"`、`MC_K=16`、
variants=`mc_greedy mc_lookahead`(center系の初期化位置感度を見たい場合は
`VARIANTS_LIST`に`mc_center mc_lookahead_center`を追加)。旧実験の
選択JSON置き場が既定(`selected_patch_index`)と違うなら`OLD_JSON_BASE=...`で指定する。
上書き防止の部分再実行:`SEEDS`/`SKIP_CALIB`/`SKIP_ZIP`/`FORCE_INIT`等はスクリプト
先頭のコメント参照。手動実行は以下のとおり。

`mc_*`バリアントの不確実性は、predictorのdropoutだけをtrainモードにして`--mc-samples`(既定16)回
forwardした時の、特徴ごとのKサンプル分散((E\[z²\]−(E\[z\])², 微小負値はclip)を特徴次元で平均した
**パッチ単位のスカラー分散**。平方根は取らない。予測値はKサンプル平均を使う。
`--mu-dropout 0.1`(既定)はIJEPAコンストラクタにdropoutを渡す設定。

```bash
# パッチ選択→再構成評価(既定: mc_greedy / mc_lookahead)
# (center系も使える: --variants mc_greedy mc_center mc_lookahead mc_lookahead_center)
python patch_selection_reconstruction_eval.py \
    --model_path models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth \
    --variants mc_greedy mc_lookahead \
    --mc-samples 16 --mc-seed 0 \
    --json_dir selected_patch_index/mcdropout/seed_0 \
    --out_dir images/rec_images_mcdropout

# calibration(不確実性=MC分散)
python calibration_eval.py \
    --model_path models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth \
    --mc-samples 16 --out_dir images/calibration_mcdropout

# 不確実性トラジェクトリ(mc_*の選択JSONをリプレイ)
python sigma_uncertainty_traj.py \
    --model_path models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth \
    --variants mc_greedy --mc-samples 16 \
    --json_dir selected_patch_index/mcdropout/seed_0 \
    --out_dir images/rec_images_mcdropout

# 旧σモデルの結果と比較(複数ptのオーバレイ; baseline系の値はモデル非依存なので流用可)
python patch_selection_reconstruction_eval.py \
    --points_pt images/rec_images_tmax32/b2/rec_error_vs_t.pt \
                images/rec_images_mcdropout/b2/rec_error_vs_t.pt \
    --out_dir images/compare_sigma_vs_mcdropout
```

#### 旧σ実験と初期ブロックを一致させる(`--init-blocks-pt`)

旧5seed σ実験と**画像単位でペアリングした比較**をする場合、`mc_*`の初期ブロックは
旧実験のものをそのまま使う。numpy乱数の消費列は`--bs`/`--variants`構成に依存するため
seed指定だけでは再現できないが、初期ブロックは旧成果物のJSON(各画像の先頭b²個の
観測patch列)から復元できる:

```bash
# 1) 旧実験の選択JSONから初期ブロックを抽出(seedごとに1回)
python make_init_blocks_pt.py \
    --json_dir selected_patch_index/seed_0 \
    --bs 2 3 4 --out init_blocks/seed_0.pt

# 2) MC評価のpseコマンドに --init-blocks-pt を付ける(上記のコマンド例に追加)
#    例: --bs 2 3 4 --init-blocks-pt init_blocks/seed_0.pt
```

指定するとbごとの共有初期ブロックがrng drawの代わりにそのファイルから読まれ、
初期化タイミングの差(初期化ラック)を消した全く同じ初期化で比較できる。
`--n`(画像数)はJSON側と照合され、不一致ならエラーで停止する。無指定なら従来
どおり`--seed`のrngでdrawする。`make_init_blocks_pt.py`はrandom-init系variantの
JSON間で先頭ブロックが一致することも検証する(旧実験のデータ整合性チェックを兼ねる)。

注意: `--model_path`の既定はσ付I-JEPA。そのモデルに`mc_*`バリアントを渡すと(μ-predictorに
dropoutがないため)エラーで停止する。σ系の結果と混ざらないよう、`--json_dir`/`--out_dir`
は上記のようにmcdropout用のディレクトリに分けること(成果物の形式はσ系と同一なので後から
`--points_pt`で比較可能)。MCコストはpredictorのK回forward分(encoderは1回のみ)。
本番(5seed・N=8000・b=2,3,4)では、旧実験と同一の`--seed/--N/--start_index/--bs`構成と
`--init-blocks-pt`(上記)を使い、4variantは1回のinvocationで実行すること(分けると
`sigma_sum_traj.pt`が上書きされる)。lookaheadは候補M×K回のpredictor forwardで重い
(旧σのlookahead選択がseedあたり30–50分 → MCのK=16では数時間/seed)。

## 成果物の保存先

### MC-Dropout学習(`revision/run.sh`)

| 成果物                             | 保存先                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------- |
| 学習済みI-JEPA(σキー込み・σ未学習) | `models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth` |
| target encoder(EMA)                | `models/mcdropout_embdim=512_epoch=300_patchsize=8/encoder.pth`        |
| 上記をまとめたzip(実行終了時に自動生成) | `result.zip`(リポジトリ直下)                                      |

### MC-Dropout評価の成果物

| 成果物 | 保存先 |
| ------ | ------ |
| 選択シーケンス | `selected_patch_index/mcdropout/seed_{s}/block_{b}_tmax_32_{mc_*}.json` |
| 再構成誤差 | `images/rec_images_mcdropout/b{b}/rec_error_vs_t.{png,csv,pt}` |
| 不確実性トラジェクトリ | `images/rec_images_mcdropout/b{b}/sigma_sum_traj.pt`(MC分散和) |
| calibration | `images/calibration_mcdropout/calibration_points.pt`(ほか既存形式と同一) |
| 旧σモデルとの比較オーバレイ | `images/compare_sigma_vs_mcdropout/rec_error_vs_t.compare.{png,pt}` |
| 旧実験から復元した初期ブロック | `init_blocks/seed_{s}.pt`(`--init-blocks-pt`で入力、成果物というより共有素材) |
