import * as tf from '@tensorflow/tfjs';

// Initialize the architecture 'structure'
const model = tf.sequential();

// Create the first layer of the structure
model.add(tf.layers.dense({ // Dense means all node3s are connected
  units: 2, // Number of nodes
  inputShape: [2], // Requires the input shape to be defined
  activation: 'sigmoid' // Activation function
}));

// Add second layer, Does not require inputShape
model.add(tf.layers.dense({
  units: 1,
  activation: 'sigmoid'
}));

// Compile the model
model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

console.log('Succesfully created model');

// Let's make some training data
const xs = tf.tensor2d([[0, 1], [1, 0]]);
const ys = tf.tensor1d([false, true]);

// // Train the model (We want to be able to let 0 = false and 1 = true)
async function train() {
    // return await model.fit(xs, ys, {epochs: 50})
}

// for(let i = 0; i < 10; i++) {
//   train().then((res) => {
//     console.log(res)
//   })
//   .catch((e) => console.log(e));
// }

// The result of my first ml application:
// Understand creatin the model
// Could not interpret training and
// Predicting data values