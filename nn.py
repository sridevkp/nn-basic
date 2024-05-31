import numpy as np
import csv

class NN():
    def __init__( self, ip, hidden, op, rate=.1 ):
        self.ip = ip
        self.hidden = hidden
        self.op = op
        self.B = [ np.random.rand( hidden, 1 ) -0.5, np.random.rand( op, 1 ) -0.5 ]
        self.W = [ np.random.rand( hidden, ip ) -0.5, np.random.rand( op, hidden ) -0.5 ]
        self.learning_rate = rate
        self.accuracy = 0

    def setW( self, weights ):
        self.W[0] = np.array( weights[0] ).reshape( self.hidden, self.ip )
        self.W[1] = np.array( weights[1] ).reshape( self.op, self.hidden )

    def setB( self, biasis ):
            self.B = [ np.array(B) for B in biasis ] 

    @staticmethod
    def load( path ):
        try:
            with open( path, "r", newline='') as f:
                reader = csv.reader(f)
                rows = [ [ float(val) for val in row ] for row in reader ]
                rows[0] = [int(i) for i in rows[0]]

                [ip, hidden, op] =  rows[0] 
                nn = NN( ip, hidden, op )
                nn.setW( rows[1:3] ) 
                nn.setB( rows[3:5] )
                return nn
        except Exception as e:
            print( e )
    
    def save( self, path ):
        with open( path, "w", newline='' ) as f:
            W = [ np.ravel(w) for w in self.W ]
            B = [ np.ravel(b) for b in self.B ]
            writer = csv.writer(f)
            writer.writerow( [self.ip, self.hidden, self.op] )
            writer.writerows( W )
            writer.writerows( B )

    def update_params( self, dW1, dB1, dW2, dB2 ):
        self.W[0] = self.W[0] - self.learning_rate* dW1
        self.W[1] = self.W[1] - self.learning_rate* dW2
        self.B[0] = self.B[0] - self.learning_rate* dB1
        self.B[1] = self.B[1] - self.learning_rate* dB2

    def fd_prop( self, A ):
        Z1 = self.W[0].dot(A) + self.B[0] 
        A1 = ReLU( Z1 )
        Z2 = self.W[1].dot(A1) + self.B[1]
        A2 = softmax( Z2 )
        return Z1, A1, Z2, A2

    def bd_prop( self, X, Z1, A1, Z2, A2, Y, m ):
        dZ2 = A2 - Y
        dW2 = 1/m * dZ2.dot( A1.T ) 
        dB2 = 1/m *np.sum( Z2 ).reshape(-1,1) 
        dZ1 = self.W[1].T.dot(dZ2) * deriv_ReLU(Z1)
        dW1 = 1/m *dZ1.dot( X.T ) 
        dB1 = 1/m *np.sum( dZ1 ).reshape(-1,1) 
        return dW1, dB1, dW2, dB2

    def get_accuracy( self, X, Y ):
        result, index = self.predict( X )
        return np.sum( result == Y) / Y.size

    def train( self, X, Y, m=1000 ):
        Z1, A1, Z2, A2 = self.fd_prop( X )
        dW1, dB1, dW2, dB2 = self.bd_prop( X, Z1, A1, Z2, A2, Y, m )
        self.update_params( dW1, dB1, dW2, dB2 )

    def predict( self, ip ):
        Z1, A1, Z2, result = self.fd_prop( A = ip )
        return result, result.argmax()

    
    

def softmax(x):
    return np.exp(x) / np.sum( np.exp(x))

def ReLU(x):
    return np.maximum(0, x)

def deriv_ReLU( x ):
    return x > 0