# Batch Generator for Deep learning 
<hr/>

you only need numpy,

you can use, one_hot for softmax 

it simple and working great for softmax, linear regression with TF

## notice 
20180623 you don't need making datax to full connect before input Batch class anymore

for example, if your datas shape are [8540, 64, 64, 3]

you can just input !  no need to make FC

## How to Test it
```python
tx = np.arange(0,8005)
ty = np.arange(0,8005)
batch = BatchGenerator(tx, ty, batch_size=30, one_hot=True, nb_classes=8005)
for i in range(1000):
    print(batch.x)
    print(np.argmax(batch.y,1))
    print('\n')
    batch.next_batch()
```


## example usage
```python
import numpy as np
import pandas as pd
train_data = pd.read_csv('fashion-mnist_train.csv', dtype='float32')
train_data = np.array(train_data)
train_Y = train_data[:,[0]]
train_X = train_data[:,1:]

...
# if one_hot = True , it makes data_y to one_hot encoding 

datas = BatchGenerator(train_X, train_Y, batch_size=100, one_hot=True, nb_classes=nb_classes)

...

for step in range(100):
    av_cost = 0
    for i in range(datas.total_batch):
        co_v, _ = sess.run([cost, train], feed_dict = {X: datas.x, Y: datas.y})
       
        datas.next_batch() # get next datas, to datas.x, datas.y
        av_cost += co_v / datas.total_batch


```
