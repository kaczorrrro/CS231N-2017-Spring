import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  out = X@W

  num_examples, num_classes = out.shape

  for i in range(num_examples):
    exped = np.exp(out[i])
    exp_sum = np.sum(exped)
    all_scores = exped/exp_sum
    for j in range(num_classes):
      dW[:, j]+=all_scores[j]*X[i]
      if j==y[i]:
        dW[:, j]-= X[i]
    loss+= -np.log( all_scores[y[i]] );

  loss/=num_examples
  dW/=num_examples

  loss += 0.5* reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  out = X@W

  num_examples, num_classes = out.shape

  exped = np.exp(out)
  exp_sum = np.sum(exped, 1).reshape((-1,1))
  all_scores = exped/exp_sum

  right_class_scores = all_scores[np.arange(num_examples), y];
  loss = -np.sum(np.log(right_class_scores))/num_examples

  all_scores[np.arange(num_examples), y] -= 1
  dW += X.T@all_scores/num_examples

  loss += 0.5* reg * np.sum(W * W)
  dW+=W*reg

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

class Softmax:
  
  def __init__(self, W):
    self.W = W

  def train(self, X, y, epochs, lr, batch_size, reg, verbose):
    num_examples = X.shape[0]
    # p = np.random.permutation(num_examples)
    # X = X[p]
    # y = y[p]
    for epoch in range(epochs):
      avg_loss = 0
      for i in range(0,num_examples, batch_size):
        batch_start = i
        batch_end = min(batch_start+batch_size, num_examples)
        x_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]
        loss, dW = softmax_loss_vectorized(self.W, x_batch, y_batch, reg)
        avg_loss+=loss
        self.W -= lr*dW
      avg_loss/= math.ceil(num_examples/batch_size)

      if epoch%verbose==0:
        print("Epoch {}, avg loss: {}".format(epoch, avg_loss))
  
  def classify(self, X):
    out = X@self.W
    return out

  def predict(self, X):
    out = X@self.W
    return np.argmax(out, axis=1)




