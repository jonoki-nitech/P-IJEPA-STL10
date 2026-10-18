"""STL10 データの読み込みユーティリティ.

サーバー側で学習データが配置されたパスが存在する場合はそちらを優先して使用し、
存在しない環境 (ローカル PC など) では ``data/`` 以下に指定した split だけを
準備して使用する。

アーカイブ (STL10 本体, 約 2.6 GB) はディスクに保存せず、ストリーミングで
必要な split のみ取り出すため、ディスク消費は取り出した .bin の分だけである。
"""

import os
import tarfile
import urllib.request

import numpy as np
from PIL import Image

# 学習データが配置されているパス (サーバー環境)
ORIG_DATA_ROOT = "/home/jonoki/research/master/byol_trial/ExtreMA_stl10/data"

# ローカル環境のデータ保存先
LOCAL_DATA_ROOT = "data"

STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"


def get_stl10_root():
    """STL10 のルートパスを返す.

    サーバー側の既存データパスが存在すればそちらを、無ければローカルの
    ``data/`` を返す。
    """
    if os.path.isdir(ORIG_DATA_ROOT):
        return ORIG_DATA_ROOT
    os.makedirs(os.path.join(LOCAL_DATA_ROOT, "stl10_binary"), exist_ok=True)
    return LOCAL_DATA_ROOT


def ensure_stl10_splits(splits=("test",)):
    """指定した split の STL10 をローカルに用意する.

    サーバー側の既存データパスが存在する場合は何もしない。
    無い場合は公式アーカイブをストリーミングで読み込み、存在しない
    split の .bin のみ ``data/stl10_binary/`` に展開する。
    """
    if os.path.isdir(ORIG_DATA_ROOT):
        return

    base = os.path.join(LOCAL_DATA_ROOT, "stl10_binary")
    os.makedirs(base, exist_ok=True)

    # アーカイブ内の命名は {split}_X.bin / {split}_y.bin (unlabeled は X のみ)
    required = []
    for s in splits:
        required.append(f"{s}_X.bin")
        if s != "unlabeled":
            required.append(f"{s}_y.bin")

    missing = [name for name in required
               if not os.path.isfile(os.path.join(base, name))]
    if not missing:
        return

    print(f"STL10 ({', '.join(missing)}) をダウンロード中 "
          f"(アーカイブは保存せずストリーミングで取り出します) ...", flush=True)
    req = urllib.request.Request(STL10_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, \
            tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name in missing:
                tf.extract(member, path=LOCAL_DATA_ROOT)
                print(f"  抽出完了: {member.name}", flush=True)
                missing.remove(name)
                if not missing:
                    break

    if missing:
        raise RuntimeError(
            f"STL10 アーカイブから以下を取り出せませんでした: {missing}")


def load_stl10(split, transform=None, target_transform=None):
    """split を指定して STL10 のデータセットを返す.

    torchvision の STL10 は整合性チェックで全 split のアーカイブを要求するため、
    必要な split のみ扱う STL10Partial を返す。
    """
    ensure_stl10_splits((split,))
    return STL10Partial(get_stl10_root(),
                        split=split,
                        transform=transform,
                        target_transform=target_transform)


class STL10Partial:
    """必要な split のみを扱う STL10 データセット (torchvision 互換の読み方).

    torchvision.datasets.STL10 と同じ形式 (uint8, W/H 転置, 1-based ラベル) で
    .bin ファイルを読むため、torchvision の存否チェック (全 split 要求) を回避する。
    """

    base_folder = "stl10_binary"

    def __init__(self, root, split="test", transform=None, target_transform=None):
        base = os.path.join(root, self.base_folder)
        x_path = os.path.join(base, f"{split}_X.bin")
        y_path = os.path.join(base, f"{split}_y.bin")
        if not os.path.isfile(x_path):
            raise RuntimeError(
                f"{x_path} が見つかりません "
                f"(data_utils.ensure_stl10_splits を先に実行してください)")

        with open(x_path, "rb") as f:
            data = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(data, (-1, 3, 96, 96))
        self.data = np.transpose(images, (0, 1, 3, 2))

        if split == "unlabeled":
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:
            with open(y_path, "rb") as f:
                self.labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # torchvision と同様に PIL Image を返す
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
