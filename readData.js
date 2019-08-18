const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const getFaceImage = require('./getFaceImage');

const basePath = './data/face-recognition';
const imgsPath = path.resolve(basePath, 'imgs');
const nameMappings = ['daryl, nikhil'];

const imgFiles = fs.readdirSync(imgsPath);

const images = imgFiles
// get absolute file path
  .map(file => path.resolve(imgsPath, file))
  // read image
  .map(filePath => cv.imread(filePath))
  // face recognizer works with gray scale images
  .map(img => img.bgrToGray())
  // detect and extract face
  .map(getFaceImage)
  // face images must be equally sized
  .map(faceImg => faceImg.resize(80, 80));

const isImageFour = (_, i) => imgFiles[i].includes('4');
const isNotImageFour = (_, i) => !isImageFour(_, i);
// use images 1 - 3 for training
const trainImages = images.filter(isNotImageFour);
// use images 4 for testing
const testImages = images.filter(isImageFour);
// make labels
const labels = [0, 0, 0, 1, 1, 1];

const eigen = new cv.EigenFaceRecognizer();
const fisher = new cv.FisherFaceRecognizer();
const lbph = new cv.LBPHFaceRecognizer();
eigen.train(trainImages, labels);
fisher.train(trainImages, labels);
lbph.train(trainImages, labels);

const runPrediction = (recognizer) => {
  testImages.forEach((img) => {
    const result = recognizer.predict(img);
    console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
  });
};


console.log('eigen:');
runPrediction(eigen);

console.log('fisher:');
runPrediction(fisher);

console.log('lbph:');
runPrediction(lbph);
