import * as tf from '@tensorflow/tfjs';

/**
* Initializes the Sentiment demo.
*/

class SentimentPredictor {
  constructor() {
    setupSentiment();
    console.log('constructor');
  }
  /**
  * Initializes the Sentiment demo.
  */
  async init(urls) {
    this.urls = urls;

    this.model = await loadHostedPretrainedModel(urls.model);
    console.log('🤖 model', this.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const sentimentMetadata = await loadHostedMetadata(this.urls.metadata);

    this.indexFrom = sentimentMetadata['index_from'];
    this.maxLen = sentimentMetadata['max_len'];
    // console.log('🤖', 'indexFrom = ' + this.indexFrom);
    // console.log('🤖', 'maxLen = ' + this.maxLen);

    this.wordIndex = sentimentMetadata['word_index'];
    this.vocabularySize = sentimentMetadata['vocabulary_size'];
    // console.log('🤖', 'vocabularySize = ', this.vocabularySize);
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
      text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');

    const sequence = inputText.map(word => {
      let wordIndex = this.wordIndex[word] + this.indexFrom;
      if (wordIndex > this.vocabularySize) {
        wordIndex = 2;
      }
      return wordIndex;
    });
    // Perform truncation and padding.
    const paddedSequence = padSequences([sequence], this.maxLen);
    const input = tf.tensor2d(paddedSequence, [1, this.maxLen]);

    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();

    return { score: score, elapsed: (endMs - beginMs) };

    // from Dan's code
    //   // Look up word indices.
    // const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    // for (let i = 0; i < inputText.length; ++i) {
    // // TODO(cais): Deal with OOV words.
    //   const word = inputText[i];
    //   inputBuffer.set(this.wordIndex[word] + this.indexFrom, 0, i);
    // }
    // const input = inputBuffer.toTensor();
    // console.log('Running inference');
    // const predictOut = this.model.predict(input);
    // const score = predictOut.dataSync()[0];
    // predictOut.dispose();
    // return { score: score };
  }
}


async function setupSentiment() {
  const HOSTED_URLS = {
    model:
          'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata:
          'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json',
  };
  const predictor = await new SentimentPredictor().init(HOSTED_URLS);
}

async function loadHostedPretrainedModel(url) {
  console.log('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadLayersModel(url);

    // fetch method try to see if works
    // const model = fetch(url)
    //   .then(res => res.json())
    //   // .then(json => console.log(json));
    //   .then((json) => {
    //     return json;
    //   });

    // We can't load a model twice due to
    // https://github.com/tensorflow/tfjs/issues/34
    // Therefore we remove the load buttons to avoid user confusion.
    return model;
  } catch (err) {
    console.error(err);
    console.log('🤖', 'Loading pretrained model failed.');
  }
}


async function loadHostedMetadata(url) {
  console.log('🤖', 'Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    console.log('🤖', 'Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    console.log('🤖', 'Loading metadata failed.');
  }
}


// setupSentiment();


module.exports = SentimentPredictor;

