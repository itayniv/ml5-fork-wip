// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Sentiment
*/


import * as tf from '@tensorflow/tfjs';

// load loader.js from local directory
import * as loader from './loader';



//load model and metadata from tensorflow (or locally?)

const HOSTED_URLS = {
  model:
  'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
  metadata:
  'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};


class PredictSentiment{

}



async function setupSentiment() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const predictor = await new SentimentPredictor().init(HOSTED_URLS);
      ui.prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  //// no local urls for now

  // if (await loader.urlExists(LOCAL_URLS.model)) {
  //   ui.status('Model available: ' + LOCAL_URLS.model);
  //   const button = document.getElementById('load-pretrained-local');
  //   button.addEventListener('click', async () => {
  //     const predictor = await new SentimentPredictor().init(LOCAL_URLS);
  //     ui.prepUI(x => predictor.predict(x));
  //   });
  //   button.style.display = 'inline-block';
  // }
  console.log("setupSentiment --> Standing by.")
  // ui.status('Standing by.');
}


setupSentiment();
