# import dependencies
import numpy as np

# create class
class LinearRegression:
    def __init__(self,  
                 learning_rate: float, 
                 random_state: int, 
                 iterations: int
                 ):
        
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.iterations = iterations
    
    def _net_input(self, X: np.array) -> float:
        return np.dot(X, self.w_) + self.b_
    
    def fit(self, X: np.array) -> None:
        # generate the random seed
        random_seed = np.random.RandomState(self.random_state)

        # initialise random weights and zero bias term
        self.w_ = [random_seed.normal(loc=0, 
                                      scale=0.01, 
                                      size=self.X.shape[1])]
        self.b_ = [0.]

        # initialise a list of loss function calculations
        self.losses_ = []

        # iterate through the iterations
        for i in range(self.iterations):
            output = self._net_input(self.X)
            loss = (self.y - output)

            self.w_ += self.learning_rate * 2.0 * X.T.dot(loss) / X.shape[0]
            self.b_ += self.learning_rate * 2.0 * loss.mean()

            self.losses_.append(loss)

            return self
        
    def predict(self, X: np.array) -> float:
        return self.net_input(X)
    
    