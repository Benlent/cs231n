import numpy as np
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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
      f = X[i].dot(W)
      f -= np.max(f)

      loss += -f[y[i]] + np.log(np.sum(np.exp(f)))
      #loss += -np.log(np.exp(f[y[i]]) / np.sum(np.exp(f)))

      for j in xrange(num_classes):
          dW[:, j] +=  1 / np.sum(np.exp(f)) * np.exp(f[j]) * X[i]
          if j == y[i]:
              dW[:, j] += -X[i]
        #   if j == y[i]:
        #       dW[:, j] += -X[i] + 1 / np.sum(np.exp(f)) * np.exp(f[j]) * X[i]
        #   else:
        #       dW[:, j] +=  1 / np.sum(np.exp(f)) * np.exp(f[j]) * X[i]

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW =  dW / num_train + reg * W

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  XW = X.dot(W)
  XW = XW - np.max(XW, axis = 1).reshape(-1, 1)

  loss = np.sum(np.log(np.sum(np.exp(XW), axis = 1)))
  loss -= np.sum(XW[xrange(num_train), list(y)])
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  dXW = np.exp(XW) * 1 / np.sum(np.exp(XW), axis = 1).reshape(-1, 1)
  dXW[xrange(num_train), list(y)] -= 1
  dW = X.T.dot(dXW)
  dW = dW / num_train + reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
