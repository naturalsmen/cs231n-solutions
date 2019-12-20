import numpy as np
from random import shuffle

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
  # cross-entropy
  # - f_yi = Xi.dot(W)
  # - q = exp(f_yi) / sum_j(exp(f_j))
  # - Li = -sum 
  num_train = X.shape[0] 
  num_classes = W.shape[1]
  
  for i in range(num_train):
    f = X[i].dot(W) # compute scores of i-th sample
    f -= np.max(f) # normalization
    scores = np.exp(f)
    
    probs = scores / np.sum(scores)
    loss += -np.log(probs[y[i]])
    
    for j in range(num_classes):
        # dL/dW = dL/df * df/dW
        # dL/df_yi = (p - 1)
        # dL/df_j = p
        # df/dW = X
        # => dL/dW_yi = (p - 1)X
        # => dL/dW_j = pX
        dW[:, j] += probs[j] * X[i]
    dW[:, y[i]] -= X[i] 
    
  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
        
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
  
  f = X.dot(W)
  f -= np.max(f, axis = 1).reshape(num_train, -1)
  scores = np.exp(f)
  
  sum = np.sum(scores, axis = 1)
  correct_class_f = f[range(num_train), y]
  
  loss += np.sum(-correct_class_f + np.log(sum))
  loss /= num_train
  loss += reg * np.sum(W * W)

  probs = scores / sum.reshape(num_train, -1)
  ind = np.zeros((num_train, num_classes))
  ind[range(num_train), y] = 1
  probs -= ind
  dW += X.T.dot(probs)

  dW /= num_train
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

