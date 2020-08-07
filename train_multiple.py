import train
import numpy as np

NUM_RUNS = 10


args = train.parse_args()

if args.task in ['dbpedia', 'yahoo']:
    NUM_RUNS = 3

test_acc = []
val_acc = []

for i in range(NUM_RUNS):
    args.seed = i+1
    cls = train.Classification(args)
    val, test = cls.run()
    val_acc.append(val)
    test_acc.append(test)

print('\n\n\n')
print(str(args))
print('val acc:', val_acc)
print('test acc:', test_acc)
print('mean val acc:', np.mean(val_acc))
print('std val acc:', np.std(val_acc, ddof=1))
print('mean test acc:', np.mean(test_acc))
print('std test acc:', np.std(test_acc, ddof=1))

with open('result.txt', 'a') as f:
    f.write(str(args))
    f.write('val acc:' + str(val_acc) + '\n')
    f.write('test acc:' + str(test_acc) + '\n')
    f.write('mean val acc:' + str(np.mean(val_acc)) + '\n')
    f.write('std val acc:' + str(np.std(val_acc, ddof=1)) + '\n')
    f.write('mean test acc:' + str(np.mean(test_acc)) + '\n')
    f.write('std test acc:' + str(np.std(test_acc, ddof=1)) + '\n\n\n')
