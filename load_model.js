const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

async function predict( input = null ) {
    console.log('1');
    let model = await tf.loadLayersModel('file://model-1a/model.json');
    model.predict( tf.tensor2d([6, 3], [1, 2]) ).print();
    console.log('3');

}

predict();
// Define a model for linear regression.



