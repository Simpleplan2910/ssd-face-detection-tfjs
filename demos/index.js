/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';
import {CLASSES} from './classes';
// import * as objectDetection from '../src';
import Stats from 'stats.js';

// const GOOGLE_CLOUD_STORAGE_DIR =
//     'https://storage.googleapis.com/tfjs-models/savedmodel/';
//const MODEL_URL = 'https://firebasestorage.googleapis.com/v0/b/weebank-c50f9.appspot.com/o/avatar%2F598c3d124a5e057137abfbe3%2Ftensorflowjs_model.pb?alt=media&token=820a28b2-1cd6-44c8-bedf-86d968bb04ea';
//const WEIGHTS_URL = 'https://firebasestorage.googleapis.com/v0/b/weebank-c50f9.appspot.com/o/avatar%2F598c3d124a5e057137abfbe3%2Fweights_manifest.json?alt=media&token=e7c27c84-ae53-4cd7-aad2-ebadefe84fc8';

const MODEL_URL = 'http://192.168.1.217:3000/savedmodel/tensorflowjs_model.pb';
const WEIGHTS_URL = 'http://192.168.1.217:3000/savedmodel/weights_manifest.json';


const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

function detectFaceInRealTime(video, model) {
  const canvas = document.getElementById('output');
  const context = canvas.getContext('2d');
  const scaleFacetor = 0.2
  // since images are being fed from a webcam
  var frameNum = 0
  // model.dispose();
  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function faceDetectionFrame() {
    // Begin monitoring code for frames per second
    stats.begin();
    frameNum ++  
    try {
      if(frameNum%10 == 0) {
        var imgscale = tf.tidy(() => {
          const pixels = tf.fromPixels(video)
          const [width, height] = [pixels.shape[0], pixels.shape[1]]          
          const img = pixels.reverse(1)      
          const imgReverse = img.expandDims(0)      
          const imgscale = imgReverse.resizeBilinear([height*scaleFacetor, width*scaleFacetor])          
          return imgscale
        })
  
        var res2
        res2  = await model.executeAsync(imgscale);  
  
        tf.dispose(imgscale)
        const count = res2[3].dataSync()[0];
        const boxes = res2[0].dataSync();
        const scores = res2[1].dataSync();
        const classes = res2[2].dataSync();
        tf.dispose(res2)

        context.clearRect(0, 0, canvas.width, canvas.height);   

        context.beginPath()
        context.save();
        context.scale(-1, 1);
        context.translate(-videoWidth, 0);
        context.drawImage(video, 0, 0, videoWidth, videoHeight);
        context.restore();

        console.log('number of detections: ', count);

        for (let i = 0; i < count; i++) {
          const min_y = boxes[i * 4] * videoHeight;
          const min_x = boxes[i * 4 + 1] * videoWidth;
          const max_y = boxes[i * 4 + 2] * videoHeight;
          const max_x = boxes[i * 4 + 3] * videoWidth;
      
          context.rect(min_x, min_y, max_x - min_x, max_y - min_y);
          context.lineWidth = 1;
          context.font = '10px Arial';
          context.strokeStyle = 'green';
          context.stroke();
          context.fillStyle = "yellow"
          context.fillText(
              scores[i].toFixed(3) + ' ' + CLASSES.find(label => label.id === classes[i]).display_name,
              min_x, min_y - 5);
        }   
      }     
           
    }
    catch(err) {  
      tf.dispose(imgscale)
      console.log(err)
    }
    console.log(tf.memory().numTensors)  
    requestAnimationFrame(faceDetectionFrame);
    // End monitoring code for frames per second
    stats.end();

    
  }

  faceDetectionFrame();
}


export async function bindPage() {
  // tf.setBackend("cpu")
  var model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  console.log('model loaded')

  document.getElementById('main').style.display = 'block';
  let video;
  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }
  setupFPS();
  detectFaceInRealTime(video, model)
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

// kick off the demo
bindPage();

