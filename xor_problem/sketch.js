let result;
let cols;
let rows;
let inp_xs;
let xs;
let ys;
let model;
let testData;
let resolution = 20;

function setup() {
  createCanvas(600, 600);
  cols = width/resolution;
  rows = height/resolution;

  let inputs = [];

  for(let i = 0; i < cols; i++) {
    for(let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      inputs.push([x1, x2]);
    }
  }

  inp_xs = tf.tensor2d(inputs);

  model = tf.sequential();
  // Start with the tensorflow model
  model.add(tf.layers.dense({activation: 'sigmoid', inputShape: [2], units: 2}));
  model.add(tf.layers.dense({activation: 'sigmoid', units: 1}));
  // Compile the model
  model.compile({optimizer: tf.train.adagrad(0.5), loss: tf.losses.meanSquaredError});

  // Training data
  xs = tf.tensor2d([
    [0, 1],
    [1, 0],
    [0, 0],
    [1, 1]]);

   ys = tf.tensor2d([
    [1],
    [1],
    [0],
    [0]
  ]);
}

async function trainModel() {
  return await model.fit(xs, ys);
}

function draw() {
  background(0);

  trainModel().then(h => console.log(h.history.loss[0]));

  let output = model.predict(inp_xs).dataSync();
  let index = 0;

    for(let i = 0; i < cols; i++) {
      for(let j = 0; j < rows; j++) {
        fill(output[index] * 255);
        rect(i * resolution, j * resolution, resolution, resolution);
        index++
      }
    }
  noLoop();
}

// SUCCESS!