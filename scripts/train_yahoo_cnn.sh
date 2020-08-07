# Random
python train_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-none

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-embed --method=embed

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-sent --method=sent

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-dense --method=dense

# nonstatic
python train_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-nonstatic-none --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-nonstatic-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-nonstatic-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=True

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-nonstatic-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=True

# static
python train_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-static-none --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-static-embed --method=embed --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-static-sent --method=sent --w2v-file=data/glove.pickle --fine-tune=False

python train_mixup_multiple.py --model=TextCNN --task=yahoo --save-path='logs/yahoo' --name=yahoo-TextCNN-static-dense --method=dense --w2v-file=data/glove.pickle --fine-tune=False
