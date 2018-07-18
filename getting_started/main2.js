// The second attempt to TensorFlow.js
// This time following along a tutorial
import * as tf from '@tensorflow/tfjs';

// Create the architecture for the ML
const model = tf.sequential();

// Add the inputs to the ML
model.add(tf.layers.dense({inputShape: [2], units: 4, activation: 'sigmoid'}));

// Add the final layer to the network
model.add(tf.layers.dense({units: 2, activation: 'sigmoid'}));

// Compile the model
model.compile({optimizer: tf.train.adam(0.1), loss: 'meanSquaredError'});

// Successfuly compiled!!
// Things to consider: Learn about options.


//========================//
 //I DONT KNOW THE BELOW//
//========================//
// ..But I'm learning


// Inputs as a tensor (ERROR)! Due to batching...
// Wrong code directly below, needs to live in an array as first element, becomes 2d tensor.

/* const inputs = tf.tensor1d([0.25, 0.92]); */

// Correct code
/* const inputs = tf.tensor2d([[0.25, 0.84]]); */ // AHAA MOMENT!
/* let outputs = model.predict(inputs) */

// Will print random results based on initial algorithm
/* outputs.print(); */

   ///// ///////// //////
  ///// // FIT // //////
 ///// ///////// //////

// similar to predict, but we provide I/O.

// Defin Input x vals
const xs = tf.tensor2d([[0.25, 0.63],
                        [0.66, 0.33],
                        [0.21, 0.54]]);

// Create the y training data
const ys = tf.tensor2d([[1, 0],
                        [0, 1],
                        [1, 0]]);
// We're inferring that values above .5 should be 1 and below should be 0.

// Train the model (ASYNC CODE).
async function trainModel() {
  for(let i = 0; i < 1000; i++) {
    const res = await model.fit(xs, ys, {epochs: 1, shuffle: true});
    console.log(res.history.loss[0]);
  }
}

tf.tidy(() => {
  // Call the train function
  trainModel().then(() => {
    // After training, let's predict!
    const testData = tf.tensor2d([
                                [0.54, 0.25],
                                [0.66, 0.66],
                                [0.21, 0.21]]); // Expected to see [0,1], [0, 0], [1, 1]
    // Create the predictions out of the model
    let predictions = model.predict(testData);

    //  Show the predictions
    predictions.print(); // Not quite, but it's happening!

    console.log('Training complete');
    console.log(tf.memory().numTensors);
  });
});

// End of the day, I understand a little more!