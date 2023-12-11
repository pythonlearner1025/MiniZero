from game.play import Play
from game.game import Go, BLACK, WHITE
from ml.proc import iterator
from ml.search import MCTS
from tqdm import tqdm
from ml.net import Net
import mlx.optimizers as optim
import mlx.nn as nn
import mlx.core as mx
import numpy as np
import copy
import time

def play_human():
  game = Play(9, BLACK)
  game.render()

def idx2move(idx,size):
  if idx == size*size:
    return -1,-1
  j = idx%size
  i = (idx-j)//size
  return i,j

def battle(newnet, oldnet, size, sims, eval_games=30, win_thresh=0.55):
  choice = [newnet, oldnet]
  ws = 0
  for i in tqdm(range(eval_games)):
    if i > 0: choice = choice[::-1]
    mcts = MCTS() 
    game = Go(size)
    while not game.game_ended():
      for _ in range(sims):
        g = Go(size)
        g.board = np.copy(game.board)
        net = choice[game.turn-1]
        mcts.search(g, net, root=True)
      pi = mcts.pi(game)
      move = idx2move(np.argmax(pi), game.size)
      game.move(move)
    winner,_,_ = game.score()
    if winner and i%2 == (winner-1)%2:
      ws += 1 
  wr = ws/eval_games
  return (newnet,wr,True) if wr > win_thresh else (oldnet,1-wr,False)

def cross_entropy(p,q):
  # for numerically stable log
  ret = mx.sum(mx.log(mx.add(p, 10**-5, stream=mx.cpu), stream=mx.cpu)*q, axis=1, stream=mx.cpu)
  return ret

def mse(a,b):
  ret = mx.square(mx.subtract(a,b),stream=mx.cpu)
  return ret

def loss_fn(net, X, y_pi, y_v, reg):
  pi,v = net(X)
  v = mx.squeeze(v,stream=mx.cpu)
  ret = mx.add(mx.subtract(mse(y_v, v), cross_entropy(y_pi, pi), stream=mx.cpu), reg, stream=mx.cpu)
  return ret.mean()

def anneal_lr(steps):
  if steps < 400: return 10**-2
  elif steps < 600: return 10**-3
  else: return 10**-4

def train_net(net, buffer, train_iters=1000, BS=32):
  mx.eval(net.parameters())
  reg = 0
  loss_and_grad_fn = nn.value_and_grad(net, loss_fn) 
  optimizer = optim.SGD(learning_rate=0.01, momentum=0.9)
  for i in tqdm(range(train_iters)):
    for X,(y_pi,y_v) in iterator(buffer, BS=BS):
      loss, grad = loss_and_grad_fn(net,X,y_pi,y_v,reg)
      optimizer.learning_rate = anneal_lr(i)
      optimizer.update(net, grad)
      # yeah idk wtf this does
      #mx.eval(net.parameters(), optimizer.state)
      print(loss)
  return net

# t goes towards 0 as play progresses.
# t = 1 for first 15% moves
def temp_dropoff(move_n, size):
  opening = size**2*0.6*0.15
  if move_n <= opening:
    return 1
  else:
    return opening/move_n

# TODO: check net works, queue tree eval (threads)
# infinitesimal temperature for exploration after first 30 moves of game
if __name__ == '__main__':
  total_epochs = 100
  self_play_games = 10
  train_iters = 50
  sims = 30
  eval_games = 30
  buffer_cutoff = int(total_epochs*self_play_games*0.125)
  win_thresh = 0.55

  blocks = 4
  size = 5
  in_features = 8*2+1

  net = Net(blocks,size,in_features)
  exp_buffer = []
  for e in range(total_epochs):
    prev_net = Net(blocks,size,in_features)
    prev_net.update(net.trainable_parameters())

    print(f"playing {self_play_games} games...")
    s = time.perf_counter()
    for i in tqdm(range(self_play_games)):
      buffer = []
      mcts = MCTS()
      game = Go(size)
      m = 1
      while not game.game_ended():
        # TODO execute on parallel threads
        for _ in range(sims):
          g = Go(size) 
          g.board = np.copy(game.board)
          mcts.search(g, net, root=True)
          #print(len(mcts.visited))
        pi = mcts.pi(game, t=temp_dropoff(m,size))
        move = idx2move(np.random.choice(np.arange(len(pi)),p=pi), game.size)
        buffer.append([copy.deepcopy(game), pi, None])
        print(f'{"B" if game.turn==1 else "W"} {move} move {m}')
        game.move(move)
        print(game.board)
        assert game.turn-1 % 2 == m % 2
        m += 1

      winner,_,_ = game.score()
      # update rewards of buffer based on winner. winner 1, loser -1 
      for j in range(len(buffer)):
        buffer[j][-1] = 1 if winner and (j%2==winner-1) else -1 
      exp_buffer += buffer

    train_net(net, exp_buffer[-buffer_cutoff:], train_iters=train_iters)

    net,wr,new_won = battle(net, prev_net, size, sims, eval_games=eval_games, win_thresh=win_thresh)
    print(f'WINNER: {"new" if new_won else "old"} | WR: {wr*100:7.2f}%')
    e = time.perf_counter()
    print(f'training will take {((e-s)*total_epochs)/(60*60):7.1f} hrs to complete')
  


  


      
    



  