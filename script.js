// script.js – complete wake word pipeline with ONNX runtime, no UI debug
import * as ort from 'onnxruntime-web';

// ----- configuration -------------------------------------------------
const SAMPLE_RATE = 16000;
const FRAMES_PER_WINDOW = 76;
const MEL_BINS = 32;
const STRIDE = 8;                // frames
const CONTEXT_EMBEDDINGS = 16;   // stack 16 embeddings
const MAX_BUFFER_SAMPLES = 48000;
const MIN_BUFFER_SAMPLES = 32000;

// internal threshold (0.5, but user never sees it)
const WAKE_THRESHOLD = 0.5;

// model files (same folder as HTML)
const MEL_MODEL_PATH = './melspectrogram.onnx';
const EMBED_MODEL_PATH = './embedding_model.onnx';
const MULTI_MODEL_PATH = './hey_Aniquen.onnx';   // <-- matches your actual filename

// ----- DOM refs ------------------------------------------------------
const statusEl = document.getElementById('status');
const bodyEl = document.body;

// ----- state ---------------------------------------------------------
let audioContext = null;
let audioBuffer = [];
let isWakeActive = false;

// ONNX sessions
let melSession = null;
let embedSession = null;
let multiSession = null;

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// ----- load ONNX models ----------------------------------------------
async function loadModels() {
    try {
        const opts = { executionProviders: ['webgpu', 'wasm'] };
        melSession = await ort.InferenceSession.create(MEL_MODEL_PATH, opts);
        embedSession = await ort.InferenceSession.create(EMBED_MODEL_PATH, opts);
        multiSession = await ort.InferenceSession.create(MULTI_MODEL_PATH, opts);
        console.log('[ANIQUEN] All ONNX models loaded.');
    } catch (err) {
        console.error('[ANIQUEN] Model load error:', err);
        statusEl.textContent = 'Model error';
        throw err;
    }
}

// ----- audio processing: mel spectrogram -----------------------------
async function computeMelSpectrogram(samples) {
    // input name is "input", shape (1, samples)
    const inputTensor = new ort.Tensor('float32', samples, [1, samples.length]);
    const results = await melSession.run({ input: inputTensor });
    // output name is "output", shape (1, 1, frames, 32) — contiguous, so
    // flattening and slicing every 32 values still gives [frame][mel] rows
    const melData = results.output.data;
    const total = melData.length;
    const frames = total / MEL_BINS;
    const normalized = new Float32Array(total);
    for (let i = 0; i < total; i++) {
        normalized[i] = melData[i] / 10 + 2;
    }
    const mel2d = [];
    for (let f = 0; f < frames; f++) {
        const start = f * MEL_BINS;
        mel2d.push(normalized.slice(start, start + MEL_BINS));
    }
    return mel2d;
}

// ----- sliding windows & embedding -----------------------------------
async function computeEmbeddings(melFrames) {
    const embeddings = [];
    const totalFrames = melFrames.length;
    if (totalFrames < FRAMES_PER_WINDOW) return embeddings;

    for (let start = 0; start <= totalFrames - FRAMES_PER_WINDOW; start += STRIDE) {
        const windowData = new Float32Array(FRAMES_PER_WINDOW * MEL_BINS);
        for (let i = 0; i < FRAMES_PER_WINDOW; i++) {
            const src = melFrames[start + i];
            for (let m = 0; m < MEL_BINS; m++) {
                windowData[i * MEL_BINS + m] = src[m];
            }
        }
        // input name is "input_1", shape (1, 76, 32, 1)
        const inputTensor = new ort.Tensor('float32', windowData, [1, FRAMES_PER_WINDOW, MEL_BINS, 1]);
        const result = await embedSession.run({ input_1: inputTensor });
        // output name is "conv2d_19", shape (1,1,1,96)
        const out = result.conv2d_19.data;
        embeddings.push(new Float32Array(out));
    }
    return embeddings;
}

// ----- multi model: wake probability ---------------------------------
async function computeWakeProbability(embeddingStack) {
    const flat = new Float32Array(16 * 96);
    for (let i = 0; i < 16; i++) {
        const emb = embeddingStack[i];
        for (let j = 0; j < 96; j++) {
            flat[i * 96 + j] = emb[j];
        }
    }
    // input name is "onnx::Flatten_0", shape (1, 16, 96)
    const inputTensor = new ort.Tensor('float32', flat, [1, 16, 96]);
    const result = await multiSession.run({ 'onnx::Flatten_0': inputTensor });
    // output name is "39", shape (1,1) — already a probability
    const score = result['39'].data[0];
    const prob = (score >= 0 && score <= 1) ? score : sigmoid(score);
    return prob;
}

// ----- main detection loop --------------------------------------------
let embeddingHistory = [];

async function processAudioChunk(newSamples) {
    audioBuffer.push(...newSamples);
    if (audioBuffer.length > MAX_BUFFER_SAMPLES) {
        audioBuffer.splice(0, audioBuffer.length - MAX_BUFFER_SAMPLES);
    }
    if (audioBuffer.length < MIN_BUFFER_SAMPLES) return;

    const chunkSize = Math.min(audioBuffer.length, 48000);
    const startIdx = audioBuffer.length - chunkSize;
    const chunk = new Float32Array(audioBuffer.slice(startIdx));

    let melFrames;
    try {
        melFrames = await computeMelSpectrogram(chunk);
    } catch (e) {
        console.warn('[ANIQUEN] mel error:', e);
        return;
    }
    if (!melFrames || melFrames.length < FRAMES_PER_WINDOW) return;

    let embeddings;
    try {
        embeddings = await computeEmbeddings(melFrames);
    } catch (e) {
        console.warn('[ANIQUEN] embed error:', e);
        return;
    }
    if (!embeddings || embeddings.length === 0) return;

    embeddingHistory.push(...embeddings);
    if (embeddingHistory.length > CONTEXT_EMBEDDINGS) {
        embeddingHistory = embeddingHistory.slice(-CONTEXT_EMBEDDINGS);
    }
    if (embeddingHistory.length < CONTEXT_EMBEDDINGS) return;

    const context = embeddingHistory.slice(-CONTEXT_EMBEDDINGS);

    let prob;
    try {
        prob = await computeWakeProbability(context);
    } catch (e) {
        console.warn('[ANIQUEN] multi error:', e);
        return;
    }

    console.log('[ANIQUEN] wake prob:', prob.toFixed(3)); // remove once confirmed working

    if (prob >= WAKE_THRESHOLD) {
        triggerWake();
    }
}

// ----- UI trigger (green flash) --------------------------------------
let wakeTimeout = null;

function triggerWake() {
    if (isWakeActive) return;
    isWakeActive = true;
    bodyEl.classList.add('wake');
    statusEl.textContent = 'Wake Word Detected!';

    if (wakeTimeout) {
        clearTimeout(wakeTimeout);
        wakeTimeout = null;
    }
    wakeTimeout = setTimeout(() => {
        bodyEl.classList.remove('wake');
        statusEl.textContent = 'Listening…';
        isWakeActive = false;
        wakeTimeout = null;
    }, 2000);
}

// ----- audio capture ---------------------------------------------------
function setupAudioProcessing(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
    });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const scaled = new Float32Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            scaled[i] = inputData[i] * 32768;
        }
        if (!isWakeActive) {
            processAudioChunk(scaled).catch(e => console.warn(e));
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
}

// ----- main init ---------------------------------------------------------
async function init() {
    try {
        await loadModels();
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: SAMPLE_RATE,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            }
        });
        setupAudioProcessing(stream);
        statusEl.textContent = 'Listening…';
        console.log('[ANIQUEN] Microphone ready, listening forever.');
        if (audioContext && audioContext.state === 'suspended') {
            await audioContext.resume();
        }
    } catch (err) {
        console.error('[ANIQUEN] Init error:', err);
        statusEl.textContent = 'Mic error';
    }
}

init();

window.addEventListener('beforeunload', () => {
    if (audioContext) {
        audioContext.close().catch(() => {});
    }
});