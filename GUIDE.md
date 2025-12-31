# DCGAN 実装ガイド

このドキュメントでは、DCGAN実装の詳細な使用方法と理論的背景を説明します。

## 📖 目次

1. [理論的背景](#理論的背景)
2. [アーキテクチャの詳細](#アーキテクチャの詳細)
3. [実装の詳細](#実装の詳細)
4. [訓練方法](#訓練方法)
5. [トラブルシューティング](#トラブルシューティング)

## 理論的背景

### 敵対的生成ネットワーク (GAN)

GANは以下の2つのコンポーネントから構成されます：

#### 1. **生成器 (Generator)**
- 目的: ランダムノイズから本物のような画像を生成
- 入力: ノイズベクトル $z \sim \mathcal{N}(0, I)$
- 出力: 生成画像 $G(z)$

#### 2. **識別器 (Discriminator)**
- 目的: 本物の画像と偽の画像を区別
- 入力: 画像 $x$ または $G(z)$
- 出力: スカラー値 $D(x) \in [0, 1]$ (本物である確率)

### ミニマックスゲーム

訓練は以下の値関数を最小化・最大化する過程です：

$$V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- **識別器**: $\max_D V(D, G)$ → 本物は1に、偽物は0に近づけたい
- **生成器**: $\min_G V(D, G)$ → 識別器を騙したい

### DCGANの工夫

元のGANより訓練を安定化させるために以下の工夫を施しています：

| 工夫 | 効果 |
|------|------|
| **バッチ正規化** | 各層の入力を正規化して内部共変量シフトを減らす |
| **LeakyReLU (識別器)** | 負の領域でも勾配が流れるため勾配消失を緩和 |
| **ReLU (生成器)** | 計算効率が良く安定した訓練を実現 |
| **ラベルスムージング** | 真のラベルを0.9に、偽のラベルを0.1に設定 |
| **転置畳み込み** | テンソルを効率的に拡大 |

## アーキテクチャの詳細

### 生成器 (Generator) の構造

```python
Generator(
  (fc): Linear(100, 8192)           # 100次元ノイズ → 4×4×512
  (tconv1): ConvTranspose2d(...)    # 4×4 → 8×8
  (bn1): BatchNorm2d(256)
  (tconv2): ConvTranspose2d(...)    # 8×8 → 16×16
  (bn2): BatchNorm2d(128)
  (tconv3): ConvTranspose2d(...)    # 16×16 → 32×32
  (bn3): BatchNorm2d(64)
  (tconv4): ConvTranspose2d(...)    # 32×32 → 64×64
)
```

**各層の役割:**

| 層 | 入力サイズ | 出力サイズ | カーネル | ストライド | 目的 |
|----|-----------|---------|--------|----------|------|
| FC | 100 | 512×4×4 | - | - | 空間情報を追加 |
| tconv1 | 512×4×4 | 256×8×8 | 4×4 | 2 | 画像拡大 |
| tconv2 | 256×8×8 | 128×16×16 | 4×4 | 2 | 画像拡大 |
| tconv3 | 128×16×16 | 64×32×32 | 4×4 | 2 | 画像拡大 |
| tconv4 | 64×32×32 | 3×64×64 | 4×4 | 2 | 最終出力 |

### 識別器 (Discriminator) の構造

```python
Discriminator(
  (conv1): Conv2d(3, 64, ...)       # 64×64 → 32×32
  (conv2): Conv2d(64, 128, ...)     # 32×32 → 16×16
  (bn2): BatchNorm2d(128)
  (conv3): Conv2d(128, 256, ...)    # 16×16 → 8×8
  (bn3): BatchNorm2d(256)
  (conv4): Conv2d(256, 512, ...)    # 8×8 → 4×4
  (bn4): BatchNorm2d(512)
  (fc): Linear(8192, 1)              # スカラー確率
)
```

**各層の役割:**

| 層 | 入力サイズ | 出力サイズ | カーネル | ストライド | 活性化 |
|----|-----------|---------|--------|----------|-------|
| conv1 | 3×64×64 | 64×32×32 | 4×4 | 2 | LeakyReLU |
| conv2 | 64×32×32 | 128×16×16 | 4×4 | 2 | LeakyReLU |
| conv3 | 128×16×16 | 256×8×8 | 4×4 | 2 | LeakyReLU |
| conv4 | 256×8×8 | 512×4×4 | 4×4 | 2 | LeakyReLU |
| FC | 512×4×4 | 1 | - | - | Sigmoid |

**注**: conv1の後はバッチ正規化なし（不安定になるため）

## 実装の詳細

### 重み初期化

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 畳み込み層: 平均0, 標準偏差0.02の正規分布
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # バッチ正規化: 平均1, 標準偏差0.02
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
```

**なぜこの初期化か？**
- 小さな初期値 (σ=0.02) で訓練の安定性を向上
- 適切な初期値が一貫性のある訓練につながる

### 損失関数

#### 識別器の損失

$$L_D = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

```python
d_loss_real = criterion(discriminator(real_images), real_labels)  # 0.9
d_loss_fake = criterion(discriminator(fake_images), fake_labels)  # 0.1
d_loss = d_loss_real + d_loss_fake
```

#### 生成器の損失

$$L_G = \mathbb{E}[\log D(G(z))]$$

```python
g_loss = criterion(discriminator(fake_images), real_labels)  # 0.9を目指す
```

### 訓練ループの流れ

```
各エポックごと:
  各バッチについて:
    
    # ステップ1: 識別器を更新
    - 本物の画像をD(x)に通す → ロス計算
    - 偽の画像をG(z)に通し、D(G(z))で分類 → ロス計算
    - 合計ロスで逆伝播、勾配更新
    
    # ステップ2: 生成器を更新
    - 新しいノイズz生成
    - G(z)で画像生成
    - D(G(z))で分類 (detach不要)
    - ロス計算、逆伝播、勾配更新
```

## 訓練方法

### 基本的な訓練

```bash
python train.py \
    --dataset cifar10 \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 0.0002
```

### パラメータチューニングガイド

#### バッチサイズ
- **小さい値 (32-64)**: メモリ効率良い、ノイズが多い
- **大きい値 (128-256)**: より安定した勾配、より多くのVRAM必要

推奨: **128** (バランスが取れている)

#### 学習率
- **0.0002**: デフォルト値、ほとんどのケースで推奨
- **0.0001**: より慎重な更新、収束が遅い
- **0.0005**: より急速な更新、不安定になる可能性

推奨: **0.0002**

#### Beta1 (Adamのモーメント係数)
- **0.5**: DCGAN論文での推奨値
- **0.9**: 標準的なディープラーニング

推奨: **0.5**

#### エポック数
- **50**: 基本的な学習確認用
- **100**: 一般的な訓練
- **200+**: 高品質な結果必要時

推奨: **100-200**

### データセット別の設定

#### CIFAR-10
```bash
python train.py \
    --dataset cifar10 \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 0.0002
```

#### MNIST
```bash
python train.py \
    --dataset mnist \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 0.0002
```

**注**: MNISTはCIFAR-10より簡単なため、エポック数を少なくできます

## トラブルシューティング

### 1. モード崩壊 (Mode Collapse)

**症状**: 生成器が限定的な種類の画像しか生成しない

**原因**:
- 学習率が高すぎる
- バッチサイズが小さすぎる
- 訓練期間が短すぎる

**対策**:
```python
# 学習率を下げる
--learning_rate 0.0001

# バッチサイズを増やす
--batch_size 256

# 訓練期間を延ばす
--num_epochs 200
```

### 2. 勾配消失 (Vanishing Gradients)

**症状**: 生成器のロスが大きくならず、画像品質が改善しない

**原因**:
- Sigmoidで勾配が飽和
- 生成器と識別器の力が大きく異なる

**対策**:
- LeakyReLUをすべての層に使用（既に実装済み）
- スペクトラル正規化を追加（オプション）

### 3. 振動する損失関数

**症状**: ロスが収束せず激しく振動

**原因**:
- 学習率が高い
- バッチサイズが小さい
- ラベルスムージング値が不適切

**対策**:
```python
# 学習率を低くする
--learning_rate 0.0001

# バッチサイズを増やす
--batch_size 256

# ラベルスムージング値の調整（train.pyの103-104行目）
real_label = 0.8  # 0.9から0.8に変更
fake_label = 0.2  # 0.1から0.2に変更
```

### 4. メモリ不足 (Out of Memory)

**症状**: CUDA out of memory エラー

**対策**:
```bash
# バッチサイズを減らす
--batch_size 64

# CPUで訓練（遅いが可能）
--device cpu

# 画像サイズを小さくする（dcgan.pyを編集）
```

## 性能評価

### 定性的評価
- 生成画像の視覚的品質
- 多様性（同じパターンの繰り返しがないか）
- リアリティ（本物に見えるか）

### 定量的評価
- **Inception Score (IS)**: 生成画像の品質と多様性を測定
- **Fréchet Inception Distance (FID)**: 本物と生成画像の分布距離

### ロス曲線の解釈

**健全な訓練の兆候:**
```
- D_loss と G_loss が比較的安定
- どちらも0に近づかない
- 振動が徐々に減少
```

**問題のある訓練の兆候:**
```
- G_loss が単調増加
- D_loss が0に張り付く
- ロスが激しく振動
```

## 参考実装例

### カスタムデータセットの使用

```python
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images = [...]  # 画像パスリスト
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, 0  # ラベルはダミー

# train.pyの get_data_loader() を修正して使用
```

### 訓練の再開

```python
# チェックポイントから再開
checkpoint_epoch = 50
gen.load_state_dict(torch.load(f'checkpoints/generator_epoch_{checkpoint_epoch}.pth'))
dis.load_state_dict(torch.load(f'checkpoints/discriminator_epoch_{checkpoint_epoch}.pth'))

# 続きから訓練
for epoch in range(checkpoint_epoch, num_epochs):
    ...
```

## 最適なハイパーパラメータ設定

| データセット | batch_size | lr | epochs | remarks |
|-------------|-----------|-----|--------|---------|
| CIFAR-10 | 128 | 0.0002 | 100-200 | 標準設定 |
| MNIST | 64 | 0.0002 | 50-100 | シンプルなため短め |
| 顔画像 | 64 | 0.0002 | 100+ | 複雑なため長め |

---

**最終更新**: 2024年12月31日
