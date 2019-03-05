// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/* eslint max-len: ["error", { "code": 180 }] */

/*
YOLO Object detection
Heavily derived from https://github.com/ModelDepot/tfjs-yolo-tiny (ModelDepot: modeldepot.io)
*/

import * as tf from '@tensorflow/tfjs';
import Video from '../utils/Video';
import { imgToTensor } from '../utils/imageUtilities';
import callCallback from '../utils/callcallback';
import CLASS_NAMES from './../utils/COCO_CLASSES';

import {
  nonMaxSuppression,
  boxesToCorners,
  head,
  filterBoxes,
  ANCHORS,
} from './postprocess';

const URL = 'https://raw.githubusercontent.com/ml5js/ml5-data-and-training/master/models/YOLO/model.json';

const DEFAULTS = {
  filterBoxesThreshold: 0.01,
  IOUThreshold: 0.4,
  classProbThreshold: 0.4,
};

// Size of the video
const imageSize = 416;

class YOLOBase extends Video {
  constructor(video, options, callback) {
    super(video, imageSize);

    this.filterBoxesThreshold = options.filterBoxesThreshold || DEFAULTS.filterBoxesThreshold;
    this.IOUThreshold = options.IOUThreshold || DEFAULTS.IOUThreshold;
    this.classProbThreshold = options.classProbThreshold || DEFAULTS.classProbThreshold;
    this.modelReady = false;
    this.isPredicting = false;
    this.ready = callCallback(this.loadModel(), callback);
    // this.then = this.ready.then;
  }

  async loadModel() {
    if (this.videoElt && !this.video) {
      this.video = await this.loadVideo();
    }
    this.model = await tf.loadModel(URL);
    this.modelReady = true;
    return this;
  }

  async detect(inputOrCallback, cb) {
    await this.ready;
    let imgToPredict;
    let callback = cb;

    if (inputOrCallback instanceof HTMLImageElement || inputOrCallback instanceof HTMLVideoElement) {
      imgToPredict = inputOrCallback;
    } else if (typeof inputOrCallback === 'object' && (inputOrCallback.elt instanceof HTMLImageElement || inputOrCallback.elt instanceof HTMLVideoElement)) {
      imgToPredict = inputOrCallback.elt; // Handle p5.js image and video.
    } else if (typeof inputOrCallback === 'function') {
      imgToPredict = this.video;
      callback = inputOrCallback;
    }

    return callCallback(this.detectInternal(imgToPredict), callback);
  }

  async detectInternal(imgToPredict) {
    await this.ready;
    await tf.nextFrame();

    this.isPredicting = true;
    const [allBoxes, boxConfidence, boxClassProbs] = tf.tidy(() => {
      const input = imgToTensor(imgToPredict, [imageSize, imageSize]);
      const activation = this.model.predict(input);
      const [boxXY, boxWH, bConfidence, bClassProbs] = head(activation, ANCHORS, 80);
      const aBoxes = boxesToCorners(boxXY, boxWH);
      return [aBoxes, bConfidence, bClassProbs];
    });

    const [boxes, scores, classes] = await filterBoxes(allBoxes, boxConfidence, boxClassProbs, this.filterBoxesThreshold);

    // If all boxes have been filtered out
    if (boxes == null) {
      return [];
    }

    const width = tf.scalar(imageSize);
    const height = tf.scalar(imageSize);
    const imageDims = tf.stack([height, width, height, width]).reshape([1, 4]);
    const boxesModified = tf.mul(boxes, imageDims);

    const [preKeepBoxesArr, scoresArr] = await Promise.all([
      boxesModified.data(), scores.data(),
    ]);

    const [keepIndx, boxesArr, keepScores] = nonMaxSuppression(
      preKeepBoxesArr,
      scoresArr,
      this.IOUThreshold,
    );

    const classesIndxArr = await classes.gather(tf.tensor1d(keepIndx, 'int32')).data();

    const results = [];

    classesIndxArr.forEach((classIndx, i) => {
      const classProb = keepScores[i];
      if (classProb < this.classProbThreshold) {
        return;
      }

      const className = CLASS_NAMES[classIndx];
      let [y, x, h, w] = boxesArr[i];

      y = Math.max(0, y);
      x = Math.max(0, x);
      h = Math.min(imageSize, h) - y;
      w = Math.min(imageSize, w) - x;

      const resultObj = {
        className,
        classProb,
        x: x / imageSize,
        y: y / imageSize,
        w: w / imageSize,
        h: h / imageSize,
      };

      results.push(resultObj);
    });

    this.isPredicting = false;
    return results;
  }
}

const YOLO = (videoOr, optionsOr, cb) => {
  let video = null;
  let options = {};
  let callback = cb;

  if (videoOr instanceof HTMLVideoElement) {
    video = videoOr;
  } else if (typeof videoOr === 'object' && videoOr.elt instanceof HTMLVideoElement) {
    video = videoOr.elt; // Handle p5.js image
  } else if (typeof videoOr === 'function') {
    callback = videoOr;
  } else if (typeof videoOr === 'object') {
    options = videoOr;
  }

  if (typeof optionsOr === 'object') {
    options = optionsOr;
  } else if (typeof optionsOr === 'function') {
    callback = optionsOr;
  }

  return new YOLOBase(video, options, callback);
};

export default YOLO;
