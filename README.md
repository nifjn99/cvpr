## 流れ

1. laionから学習データ取得
2. MS COCO等の学習データ取得
3. 学習データ整形
4. 事前学習
5. ファインチューニング

- ## laionから学習データ取得
  - 実行すること
    - 以下からlaionのメタデータファイルを取得してnifty/laion_meta_file/に格納する
    - ファイルは"part-00000-xxxxxxx.parquet"という形式で32個あるが、part-00000からpart-00009までの10個を格納する（kaggleアカウントが必要なので持っていない場合などはご連絡ください）
      - https://www.kaggle.com/datasets/romainbeaumont/laion400m
    - nifty/src/make_laion_data.pyを実行
  - 結果
    - nifty/data/laion/にsplit0, split1, ... , split9と10個のディレクトリが作られる
    - 各splitXの中には1Mの画像とキャプションのペアが作成される（例 0.pngと0.txt）
   
- ## MS COCO等の学習データ取得
  - 実行すること
    -  以下のサイトにあるリンクからMS COCO, CelebA-HQ, CUBの学習データとテストデータを取得して、nifty/data/に格納する
    -  https://github.com/drboog/Lafite
 
- ## 学習データ整形
  - 実行すること
    - nifty/Lafite/dataset_tool.pyを実行する
    - 実行方法：`python dataset_tool.py --source=./../data/split0/ --dest=./../data/split0.zip --width=256 --height=256 --transform=center-crop`
    - split0からsplit9までの各splitで実行する
  - 結果
    - nifty/data/にsplitX.zipが作成される（これが実際に学習に用いることができるデータ）
   
- ## 事前学習
  - 実行すること
    - nifty/Lafite/train_lr_split_data5M.pyとtrain_lr_split_data10M.pyを実行する
    - 実行方法
      - キャプションなし：`python train_lr_split_data5M.py --gpus=1 --outdir=./../outputs/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./../data/birds_test_clip.zip --test_data=./../data/birds_test_clip.zip --mixing_prob=1.0 --snap=500`
      - キャプションあり：`python train_lr_split_data5M.py --gpus=1 --outdir=./../outputs/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./../data/birds_test_clip.zip --test_data=./../data/birds_test_clip.zip --mixing_prob=0.0 --snap=500`
    - 学習したパラメータは定期的に"network-snapshot-XXXXXX.pkl"に保存され、XXXXXXの部分は学習が進むにつれて大きな値となる。"network-snapshot-010000.pkl"が保存されるまで学習を行う。
  - 結果
    - nifty/outputs/に学習結果が保存される
  - 補足
    - 引数で指定しているbirds_clip_test.zipは学習に用いられず、実際には用意した学習データlaion/splitX.zipが用いられる

- ## ファインチューニング
  - 実行すること
    - nifty/Lafite/train_lr.pyを実行する
    - 以下の--resumeの引数には事前学習したパラメータのパスを指定する（network-snapshot-010000.pkl）
    - MS COCO（10%）
      - キャプションで学習：`python train_lr.py --gpus=1 --outdir=./../outputs/mscoco/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./../data/COCO2014_train_CLIP_ViTB32.zip --test_data=./../data/COCO2014_val_CLIP_ViTB32.zip --mixing_prob=0.0 --snap=100 --subset=8261 --d_lr=0.001 --resume={事前学習パラメータのパス}`
    - CelebA-HQ（10%）
      - キャプションで学習：`python train_lr.py --gpus=1 --outdir=./../outputs/celeba/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./../data/celeba_train_clip.zip --test_data=./../data/celeba_test_clip.zip --mixing_prob=0.0 --snap=40 --subset=2400 --d_lr=0.001 --g_lr=0.001 --resume={事前学習パラメータのパス}`
    - CUB（10%）
      - キャプションで学習：`python train_lr.py --gpus=1 --outdir=./../outputs/cub/ --temp=0.5 --itd=10 --itc=10 --gamma=10 --mirror=1 --data=./../data/birds_train_clip.zip --test_data=./../data/birds_test_clip.zip --mixing_prob=0.0 --snap=10 --subset=885 --d_lr=0.001 --g_lr=0.001 --resume={事前学習パラメータのパス}`
  - 学習が収束した時点のFIDで評価
    - FIDは、結果が出力されるディレクトリ内の"metric-fid50k_full.jsonl"に記載されている
