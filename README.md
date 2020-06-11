# chainer-pix2pixHD
* Chainer implementation of pix2pixHD. (https://github.com/NVIDIA/pix2pixHD)
* This version does not (yet) implement instance labels or VGG feature loss.
（このバージョンでは，まだインスタンスラベルやVGG特徴量の損失は実装されていません．）

# Example／推定結果の一例および比較
<table border="0">
<tr>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/37_lbl.png"></td>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/440_lbl.png"></td>
</tr>
</table>

<table border="0">
<tr>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/37_gen.png"></td>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/440_gen.png"></td>
</tr>
</table>

<table border="0">
<tr>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/37_img.png"></td>
<td><img src="https://github.com/rcalland/chainer-pix2pixHD/blob/master/images/440_img.png"></td>
</tr>
</table>

* 画像は，上段からラベルマップ（なぜ黒い？），Pix2pixかCRNの結果，Pix2pixHDの結果である．
 
# Setup／初期設定
* `pip install -r requirements.txt`
* `conda install -c menpo opencv3=3.1.0`
* `pip install -U git+https://github.com/mcordts/cityscapesScripts.git (in order to use the cityscapes dataset properly)`

# Usage／使用方法
* To reproduce the results on cityscapes at `512x1024` resolution, first pretrain the global generator at `256x512`:
（解像度 `512x1024` で街並みを再現するためには，まず解像度 `256x512` で大域ジェネレータを事前学習する必要がある．）
* `python tools/train.py -g <gpu> -G GlobalGenerator -o out/pretrained_global/ --config configs/cityscapes_256.json`
* Then tune the full model at `512x1024` using the results of the pretraining:
（次に，事前学習の結果を用いて， `512x1024` でフルモデルをチューニングする．）
* `python tools/train.py -g <gpu> -G MultiScaleGenerator -o out/total/ --global_generator_model out/pretrained_global/generator_model_200 --fix_global_num_epochs 10 --config configs/cityscapes_512.json`
 
* Note that you will require a GPU with at least 16Gb of VRAM to train for image size `512x1024`. The results shown here were trained on a Tesla P100.  
（画像サイズ `512x1024` を学習するには，少なくとも 16Gb の VRAM を持つ GPU が必要になることに注意してください．ここに示された結果は，Tesla P100で学習したものです． ）

# Multi-GPU training／複数のGPUで学習したい場合
* You can use [ChainerMN](https://github.com/chainer/chainermn) for multi-GPU training. Please install ChainerMN in order to use it. The following command illustrates usage:
（マルチGPU学習には[ChainerMN](https://github.com/chainer/chainermn)を使用することができます．使用するためには，ChainerMNをインストールしてください．以下のコマンドで使用方法を説明します）
* `mpiexec -n 4 python tools/train.py -g 0 1 2 3 ...`
* where we have specified 4 workers and the individual GPU ids note that this is the equivalent of having a batchsize of 4. 
（ここでは4つのワーカーと個々のGPU IDを指定していますが，これはバッチサイズが4であることと同等であることに注意してください．）
* The argument `-b` will specify the batchsize of each worker, so the effective batchsize is multiplied by the number of workers.
（引数 `-b` は各ワーカーのバッチサイズを指定するので，有効なバッチサイズはワーカーの数を掛けたものになります．）
