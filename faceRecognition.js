const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const _ = require('lodash');

if (!cv.xmodules.face) {
  throw new Error('exiting: opencv4nodejs compiled without face module');
}

const basePath = 'data/face-recognition';
const imgsPath = path.resolve(basePath, 'imgs');
const nameMappings = ['nikhil', 'daryl'];

const imgFiles = fs.readdirSync(imgsPath);

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
const getFaceImage = (grayImg) => {
  const faceRects = classifier.detectMultiScale(grayImg).objects;
  if (!faceRects.length) {
    throw new Error('failed to detect faces');
  }
  return grayImg.getRegion(faceRects[0]);
};

const images = imgFiles
  .map(file => path.resolve(imgsPath, file))
  .map(filePath => cv.imread(filePath))
  .map(img => img.bgrToGray())
  .map(getFaceImage)
  .map(faceImg => faceImg.resize(80, 80));

const trainImages = images;
const labels = imgFiles
 .map(file => nameMappings.findIndex(name => file.includes(name)));

const testImagesPath = 'data/testImages';
const testImgsPath = path.resolve(testImagesPath, 'imgs');
const testImgFiles = fs.readdirSync(testImgsPath);

console.log(testImgFiles);
const formattedTestImages = testImgFiles
  .map(file => path.resolve(testImgsPath, file))
  .map(filePath => cv.imread(filePath))
  .map(img => img.bgrToGray())
  .map(getFaceImage)
  .map(faceImg => faceImg.resize(80, 80));

const runPrediction = (recognizer) => {
  formattedTestImages.forEach((img) => {
    const result = recognizer.predict(img);
    console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
  });
};

const eigen = new cv.EigenFaceRecognizer();
const fisher = new cv.FisherFaceRecognizer();
const lbph = new cv.LBPHFaceRecognizer();
eigen.train(trainImages, labels);
fisher.train(trainImages, labels);
lbph.train(trainImages, labels);

console.log('eigen:');
runPrediction(eigen);

console.log('fisher:');
runPrediction(fisher);

console.log('lbph:');
runPrediction(lbph);
