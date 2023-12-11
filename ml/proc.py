import numpy as np
import mlx.core as mx
from game.game import BLACK
import random

# TODO test net, reg variables, buffer scheduling, checkpoint save net 
def make_binary(board,turn):
  allies = np.argwhere(board == turn)
  newboard = np.zeros((len(board),len(board)))
  newboard[[tup[0] for tup in allies],[tup[1] for tup in allies]] = 1
  return newboard

def features(game,ctx_window=8):
  # 8 of C, 8 of opp, interleaved
  # [Xt,Yt,Xt-1,Yt-1...C]  
  features = 2*ctx_window+1
  feats = [make_binary(his[2],his[0]) for his in game.history[::-1][:features-1]] 
  feats += [np.zeros((game.size,game.size)) for _ in range(features-1-len(feats))]
  feats += [np.ones((game.size,game.size)) if game.turn == BLACK else np.zeros((game.size,game.size))]
  # stack feats
  feats = mx.transpose(mx.array(np.stack(feats)), axes=[1,2,0])
  assert feats.shape == [game.size, game.size, features]
  return feats

def iterator(buffer, BS=32):
  random.shuffle(buffer)
  for i in range(0, len(buffer), BS):
    X,Y1,Y2 = [],[],[]
    for j in range(i,i+BS):
      if j >= len(buffer): break
      X.append(features(buffer[j][0])) 
      Y1.append(buffer[j][1])
      Y2.append(np.array(buffer[j][2]))
    # stack X, Y
    yield mx.array(np.stack(X)), (mx.array(np.stack(Y1)), mx.array(np.stack(Y2)))