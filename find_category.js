const tf = require('@tensorflow/tfjs');
const data = require('./data.json');
require('@tensorflow/tfjs-node');


 
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
 
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});



  const trainingDataInput = data.map(item => ({
    input: item.text,
  }));

  const trainingDataOutput = data.map(item => ({
    input: item.category,
  }));

  const countData = data.length ;
// // Generate some synthetic data for training.
const xs = tf.tensor2d(trainingDataInput, [countData, 1]);
const ys = tf.tensor2d(trainingDataOutput, [countData, 1]);
 
// Train the model using the data.
model.fit(xs, ys, {epochs: 3}).then(() => {
    const predicted = model.predict( tf.tensor2d(['i love computers'], [1, 1]) );
    console.log("resultat is : ", predicted);
});


// // Train the model using the data.
// model.fit(xs, ys, {epochs: 3000}).then(() => {
//     // Use the model to do inference on a data point the model hasn't seen before:
//     model.save('file://./model-1a');
// });

