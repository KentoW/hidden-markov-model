# hidden-markov-model
##概要
隠れマルコフモデル(hedden markov model)をPythonで実装
無限隠れマルコフモデル(hedden markov model)をPythonで実装
##hidden_markov_model.pyの使い方(隠れマルコフモデル)
```python
# Sample code.
from hidden_markov_model import HMM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
K = 10          # 隠れ変数の数
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

hmm = HMM("data.txt")
hmm.set_param(alpha, beta, K, N, converge)
hmm.learn()
hmm.output_model()
```
##infinite_hidden_markov_model.pyの使い方(隠れマルコフモデル)
```python
# Sample code.
from infinite_hidden_markov_model import IHMM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

ihmm = IHMM("data.txt")
ihmm.set_param(alpha, beta, N, converge)
ihmm.learn()
ihmm.output_model()
```
##入力フォーマット
1単語をスペースで分割した1行1文形式  
先頭に#(シャープ)記号を入れてコメントアウトを記述可能
```
# 文1
単語1 単語2 単語3 ...
# 文2
単語10 単語11 単語11 ...
...
```
例として[Wiki.py](https://github.com/KentoW/wiki)を使用して収集した アニメのあらすじ文をdata.txtに保存
