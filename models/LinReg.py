class LinearRegression:
    def __init__(self, params):
        self.params = params
        print(">>>Building LinReg model...")

    def forward(self, x):
        return x.dot(self.params).flatten()

    def backwards(self, x, y, y_estimate):
        return -(1.0 / len(y)) * (y.flatten() - y_estimate).dot(x)
