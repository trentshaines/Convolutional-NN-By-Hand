class Sequential:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.loss = None
        self.loss_prime = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def fit(self, X_train, Y_train, epochs=1000, alpha=0.01, verbose=True):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X_train, Y_train):
                output = self.predict(x)
                error += self.loss(y, output)
                gradient = self.loss_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward_prop(gradient, alpha)

            error /= len(Y_train)
            if verbose:
                print(f"Epoch: {epoch + 1}/{epochs}, Training Error = {error}")
