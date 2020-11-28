// activation function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
    // return (sigmoid(x) * 1 - sigmoid(x))
    return y * (1 - y);
}

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.learningRate = 0.3;
        if (inputNodes instanceof NeuralNetwork) {
            let a = inputNodes;
            this.inputNodes = a.inputNodes;
            this.hiddenNodes = a.hiddenNodes;
            this.outputNodes = a.outputNodes;

            this.weights_ih = a.weights_ih.copy();
            this.weights_ho = a.weights_ho.copy();

            this.hiddenBias = a.hiddenBias.copy();
            this.outputBias = a.outputBias.copy();
        } else {

            this.inputNodes = inputNodes;
            this.hiddenNodes = hiddenNodes;
            this.outputNodes = outputNodes;

            this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
            this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
            // set initial weights
            this.weights_ih.randomize();
            this.weights_ho.randomize();

            // create a bias for every node
            this.hiddenBias = new Matrix(this.hiddenNodes, 1);
            this.outputBias = new Matrix(this.outputNodes, 1);
            this.hiddenBias.randomize();
            this.outputBias.randomize();
            

        }
    }

    feedforward(input_array) {
        // generating hidden outputs
        let input = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, input);
        hidden.add(this.hiddenBias);
        hidden.map(sigmoid);

        // generating outputs output
        let output = Matrix.multiply(this.weights_ho, hidden);
        output.add(this.outputBias);
        output.map(sigmoid);

        // returning output as array
        return output.toArray();
    }
    setLearningRate(lr) {
        this.learningRate = lr;
    }
    predict(inputs) {
        return (this.feedforward(inputs));
    }

    train(input_array, target_array) {
        // generating hidden outputs
        let input = Matrix.fromArray(input_array);
        let hidden = Matrix.multiply(this.weights_ih, input);
        hidden.add(this.hiddenBias);
        hidden.map(sigmoid);

        // generating outputs output
        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.outputBias);
        outputs.map(sigmoid);

        // convert array to matrix object
        // outputs = Matrix.fromArray(outputs);
        let targets = Matrix.fromArray(target_array);

        // calculate the error
        // error = targets - outputs
        let outputErrors = Matrix.subtract(targets, outputs);

        // Calculate gradient
        let gradients = Matrix.map(outputs, dsigmoid);
        gradients.multiply(outputErrors);
        gradients.multiply(this.learningRate);

        // calc deltas
        let hidden_t = Matrix.transpose(hidden);
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_t);

        // Adjust the weights by deltas 
        this.weights_ho.add(weights_ho_deltas);
        // Adjust bias by its deltas (its deltas is the same as the gradient)
        this.outputBias.add(gradients);



        // Calculate hidden layer errors
        let weights_ho_transposed = Matrix.transpose(this.weights_ho);
        let hiddenErrors = Matrix.multiply(weights_ho_transposed, outputErrors);


        // calc hidden grad
        let hidden_gradients = Matrix.map(hidden, dsigmoid);
        hidden_gradients.multiply(hiddenErrors);
        hidden_gradients.multiply(this.learningRate);

        // calc deltas
        let input_t = Matrix.transpose(input);
        let weights_ih_deltas = Matrix.multiply(hidden_gradients, input_t);



        // Adjust the weights by deltas 
        this.weights_ih.add(weights_ih_deltas);
        // Adjust bias by its deltas (its deltas is the same as the gradient)
        this.hiddenBias.add(hidden_gradients);
    }
    // Adding function for neuro-evolution
    copy() {
        return new NeuralNetwork(this);
    }

    // Accept an arbitrary function for mutation
    mutate(func) {
        this.weights_ih.map(func);
        this.weights_ho.map(func);
        this.bias_h.map(func);
        this.bias_o.map(func);
    }
}