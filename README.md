# char-cnn-classifier

CNN over characters for list of expressions (words) classification. I wrote this code for Kaggle's https://www.kaggle.com/c/whats-cooking competition. This model achieves decent accuracy (top 5% score in Kaggle's leaderboard).


### Requirements

You need torch to run this code. You will also need a bunch of packages.

```
$ luarocks install nngraph
$ luarocks install csvigo
$ luarocks install optim
$ luarocks install nn
$ luarocks install pprint
```

### How to run the code

Train:

```
luajit main.lua -source ../train.json
```

Predict:

```
luajit main.lua -restore 1 -evaluate 2 -source ../test.json
```
