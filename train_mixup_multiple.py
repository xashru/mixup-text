import train_embed_mixup as train
import numpy as np

NUM_RUNS = 10


args = train.parse_args()

test_acc = []
val_acc = []

for i in range(NUM_RUNS):
    args.seed = i+1
    val, test = train.main(args)
    val_acc.append(val)
    test_acc.append(test)

print('\n\n\n')
print(str(args))
print('val acc:', val_acc)
print('test acc:', val_acc)
print('mean val acc:', np.mean(val_acc))
print('std val acc:', np.std(val_acc, ddof=1))
print('mean test acc:', np.mean(test_acc))
print('std test acc:', np.std(test_acc, ddof=1))
