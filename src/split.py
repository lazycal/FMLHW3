sex={'M': [1,0,0], 'F': [0,1,0], 'I': [0,0,1]}
res=[]
with open('./abalone.data') as fin:
    for line in fin.readlines():
        a=line.strip().split(',')
        a=[a[-1]]+sex[a[0]]+a[1:-1]
        res.append(f"{int(int(a[0])>9)} "+" ".join(
            [f"{i}:{a[i]}" for i in range(1, len(a))]))
with open('./train', 'w') as fout:
    fout.write("\n".join(res[:3133]))
with open('./test', 'w') as fout:
    fout.write("\n".join(res[-1044:]))

import random
with open('./train') as fin:
    lines = fin.readlines()
    print(len(lines))
nfold = 10
n = len(lines)
assert n == 3133
idx = list(range(n))
random.seed(0)
random.shuffle(idx)
# print(idx)
for i in range(nfold):
    step = n/nfold
    st, ed = int(i*step), int((i+1)*step)
    fold_val = [lines[j] for j in idx[st:ed]]
    fold_train = [lines[j] for j in (idx[:st]+idx[ed:])]
    assert len(fold_train) + len(fold_val) == n
    with open(f'./train.{i}', 'w') as fout:
        fout.write(''.join(fold_train))
    with open(f'./val.{i}', 'w') as fout:
        fout.write(''.join(fold_val))