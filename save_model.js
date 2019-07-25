const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');


 
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [2]}));
 
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
// Generate some synthetic data for training.
const xs = tf.tensor2d([[1, 1], [2, 2], [3, 3], [1, 2], [2, 3], [3, 4], [4, 5]], [7, 2]);
const ys = tf.tensor2d([2, 4, 6, 3, 5, 7, 9], [7, 1]);
 


// Train the model using the data.
model.fit(xs, ys, {epochs: 3000}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.save('file://./model-1a');
});

