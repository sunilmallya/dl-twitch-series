'''
A helper class that can be leveraged for most RNN regression tasks with Gluon

@sunilmallya
'''


import mxnet as mx
import numpy as np
import math

from mxnet import nd, autograd
from sklearn.metrics import mean_squared_error


def get_data(batch, iter_type):
    ''' get data and label from the iterator/dataloader '''
    if iter_type == 'mxiter':
        X_train = batch.data[0].as_in_context(ctx)
        Y_train = batch.label[0].as_in_context(ctx)
    else:
        X_train = batch[0].as_in_context(ctx)
        Y_train = batch[1].as_in_context(ctx)

    return X_train, Y_train
    
class BaseRNNRegressor(mx.gluon.Block):
    def __init__(self, ctx):
        super(BaseRNNRegressor, self).__init__()
        self.ctx = ctx
        self.rnn = None
        self.rnn_size = None
        
    #@override 
    def build_model(self, rnn_type='lstm', rnn_size=128, n_layer=1, n_out=1):
        self.rnn_size = rnn_size
        self.n_layer = n_layer
        self.net = mx.gluon.rnn.LSTM(rnn_size, n_layer, 'NTC')
        self.output = mx.gluon.nn.Dense(n_out)

    #@override 
    def forward(self, inp, hidden):
        rnn_out, hidden = self.net(inp, hidden)
        #simplify
        logits = self.output(rnn_out.reshape((-1, self.rnn_size)))
        return logits, hidden
        
    def detach(self, arrs):
        return [arr.detach() for arr in arrs]

    def compile_model(self, optimizer='adam', lr=1E-3):
        self.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.loss = mx.gluon.loss.L1Loss()
        self.optimizer = mx.gluon.Trainer(self.collect_params(), 
                                    optimizer, {'learning_rate': lr})


    def evaluate_accuracy(self, data_iterator, metric='mae', iter_type='mxiter'):
        met = mx.metric.MAE()
        for i, batch in enumerate(data_iterator):
            data, label = get_data(batch, iter_type)
            preds = self.net(data)
            met.update(labels=label, preds=preds)
        return met.get()
    
    def fit(self, train_data, test_data, epochs):
        moving_loss = 0.
        train_loss = []
        val_loss = []
        iter_type = 'numpy'
        
        # Can take MX NDArrayIter, or DataLoader
        if isinstance(train_data, mx.io.NDArrayIter):
            train_iter = train_data
            #total_batches = train_iter.num_data // train_iter.batch_size
            test_iter = test_data
            iter_type = 'mxiter'

        elif isinstance(train_data, list):
            if isinstance(train_data[0], np.ndarray) and isinstance(train_data[1], np.ndarray):
                X, y = train_data[0], train_data[1]
                #print type(test_data[0])
                #if isinstance(test_data[0], np.ndarray):
                #    raise ValueError("need test array to be numpy array")
                tX, ty = test_data[0], test_data[1]
                train_iter = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X, y), 
                                    batch_size=batch_size, shuffle=True, last_batch='discard')
                test_iter = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(tX, ty), 
                                    batch_size=batch_size, shuffle=False, last_batch='discard')
                #total_batches = len(X) // batch_size
        else:
            raise "pass mxnet ndarray or numpy array"

        print "data type:", type(train_data), type(test_data), iter_type

        init_state = mx.nd.zeros((1, batch_size, self.rnn_size), ctx)
        hidden = [init_state] * 2
        
        for e in range(epochs):
            if isinstance(train_iter, mx.io.NDArrayIter): train_iter.reset()
            yhat = []
            for i, batch in enumerate(train_iter):
                data, label = get_data(batch, iter_type)
                with autograd.record(train_mode=True):
                    Y_pred, hidden = self.forward(data, hidden)
                    hidden = self.detach(hidden)
                    loss = self.loss(Y_pred, label) 
                    yhat.extend(Y_pred)
                loss.backward()                                        
                self.optimizer.step(batch_size)

                if i == 0:
                    moving_loss = nd.mean(loss).asscalar()
                else:
                    moving_loss = .99 * moving_loss + .01 * mx.nd.mean(loss).asscalar()
            train_loss.append(moving_loss)
            # TODO: add prediction?
            
            test_err = self.evaluate_accuracy(test_iter, iter_type=iter_type)
            val_loss.append(test_err[1])
            print("Epoch %s. Loss: %.10f Test MAE: %s" % (e, moving_loss, test_err))
        return train_loss, val_loss
