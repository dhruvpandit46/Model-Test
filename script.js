// script.js – complete wake word pipeline with ONNX runtime
import * as ort from 'onnxruntime-web';

// ----- configuration -------------------------------------------------
const SAMPLE_RATE = 16000;
const FRAMES_PER_WINDOW = 76;
const MEL_BINS = 32;
const STRIDE = 8;
const CONTEXT_EMBEDDINGS = 16;

const MAX_BUFFER_SAMPLES = 48000;
const MIN_BUFFER_SAMPLES = 32000;

const WAKE_THRESHOLD = 0.3;
const CYCLE_HISTORY_LEN = 4;

// target RMS level we normalize every chunk to, regardless of device/gain
// quirks. Chosen to roughly match the loudness range where the model was
// tested successfully (a clearly-spoken word, not whisper-quiet).
const TARGET_RMS = 600;
const MIN_RMS_TO_NORMALIZE = 30;  // avoid amplifying near-total silence into pure noise
const MAX_GAIN = 8;               // cap how much we're willing to boost a quiet signal

const MEL_MODEL_PATH = './melspectrogram.onnx';
const EMBED_MODEL_PATH = './embedding_model.onnx';
const MULTI_MODEL_PATH = './hey_Aniquen.onnx';

const statusEl = document.getElementById('status');
const bodyEl = document.body;

let audioContext = null;
let audioBuffer = [];
let isWakeActive = false;
let isProcessing = false;
let deviceSampleRate = SAMPLE_RATE;

let melSession = null;
let embedSession = null;
let multiSession = null;

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// ----- resampling (for mobile devices that ignore requested sampleRate) --
function resampleTo16k(input, inputRate) {
    if (inputRate === SAMPLE_RATE) return input;
    const ratio = inputRate / SAMPLE_RATE;
    const newLength = Math.round(input.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const srcIndex = i * ratio;
        const idx0 = Math.floor(srcIndex);
        const idx1 = Math.min(idx0 + 1, input.length - 1);
        const frac = srcIndex - idx0;
        result[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
    }
    return result;
}

// ----- manual gain normalization ------------------------------------
// Browser AGC behaves very differently across devices (some mobile
// devices ramp gain up aggressively even during silence, drowning
// speech in amplified background noise). This applies our OWN
// consistent normalization so the model always sees similar loudness,
// regardless of device quirks.
function normalizeGain(samples) {
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
    const rms = Math.sqrt(sumSq / samples.length);

    if (rms < MIN_RMS_TO_NORMALIZE) {
        // essentially silence — don't amplify noise floor
        return { samples, rms, gain: 1 };
    }

    let gain = TARGET_RMS / rms;
    gain = Math.max(0.1, Math.min(MAX_GAIN, gain)); // clamp both directions

    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
        out[i] = samples[i] * gain;
    }
    return { samples: out, rms, gain };
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

// ----- mel spectrogram -------------------------------------------------
async function computeMelSpectrogram(samples) {
    const inputTensor = new ort.Tensor('float32', samples, [1, samples.length]);
    const results = await melSession.run({ input: inputTensor });
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

// ----- embeddings --------------------------------------------------------
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
        const inputTensor = new ort.Tensor('float32', windowData, [1, FRAMES_PER_WINDOW, MEL_BINS, 1]);
        const result = await embedSession.run({ input_1: inputTensor });
        const out = result.conv2d_19.data;
        embeddings.push(new Float32Array(out));
    }
    return embeddings;
}

// ----- multi model -------------------------------------------------------
async function computeWakeProbability(embeddingStack) {
    const flat = new Float32Array(16 * 96);
    for (let i = 0; i < 16; i++) {
        const emb = embeddingStack[i];
        for (let j = 0; j < 96; j++) {
            flat[i * 96 + j] = emb[j];
        }
    }
    const inputTensor = new ort.Tensor('float32', flat, [1, 16, 96]);
    const result = await multiSession.run({ 'onnx::Flatten_0': inputTensor });
    const score = result['39'].data[0];
    return (score >= 0 && score <= 1) ? score : sigmoid(score);
}

// ----- main detection loop --------------------------------------------
let embeddingHistory = [];
let cycleMaxHistory = [];
const EMBEDDING_HISTORY_MAX = 40;

async function processAudioChunk() {
    if (isProcessing) return;
    isProcessing = true;

    try {
        if (audioBuffer.length < MIN_BUFFER_SAMPLES) return;

        const chunkSize = Math.min(audioBuffer.length, MAX_BUFFER_SAMPLES);
        const startIdx = audioBuffer.length - chunkSize;
        const rawChunk = new Float32Array(audioBuffer.slice(startIdx));

        const { samples: chunk, rms, gain } = normalizeGain(rawChunk);

        let melFrames;
        try {
            melFrames = await computeMelSpectrogram(chunk);
        } catch (e) {
            console.warn('[ANIQUEN] mel error:', e);
            return;
        }
        if (!melFrames || melFrames.length < FRAMES_PER_WINDOW) return;

        let newEmbeddings;
        try {
            newEmbeddings = await computeEmbeddings(melFrames);
        } catch (e) {
            console.warn('[ANIQUEN] embed error:', e);
            return;
        }
        if (!newEmbeddings || newEmbeddings.length === 0) return;

        embeddingHistory.push(...newEmbeddings);
        if (embeddingHistory.length > EMBEDDING_HISTORY_MAX) {
            embeddingHistory = embeddingHistory.slice(-EMBEDDING_HISTORY_MAX);
        }
        if (embeddingHistory.length < CONTEXT_EMBEDDINGS) return;

        let cycleMaxProb = 0;
        let triggered = false;
        const lastPossibleStart = embeddingHistory.length - CONTEXT_EMBEDDINGS;
        const checkFrom = Math.max(0, lastPossibleStart - newEmbeddings.length);
        for (let start = checkFrom; start <= lastPossibleStart; start++) {
            const context = embeddingHistory.slice(start, start + CONTEXT_EMBEDDINGS);
            let prob;
            try {
                prob = await computeWakeProbability(context);
            } catch (e) {
                console.warn('[ANIQUEN] multi error:', e);
                continue;
            }
            if (prob > cycleMaxProb) cycleMaxProb = prob;
            if (prob >= WAKE_THRESHOLD) {
                triggered = true;
                break;
            }
        }

        cycleMaxHistory.push(cycleMaxProb);
        if (cycleMaxHistory.length > CYCLE_HISTORY_LEN) {
            cycleMaxHistory = cycleMaxHistory.slice(-CYCLE_HISTORY_LEN);
        }
        const recentRollingMax = Math.max(...cycleMaxHistory);

        console.log(
            '[ANIQUEN] raw rms:', rms.toFixed(1),
            ' gain applied:', gain.toFixed(2),
            ' cycle max prob:', cycleMaxProb.toFixed(3),
            ' rolling max:', recentRollingMax.toFixed(3)
        );

        if (triggered || recentRollingMax >= WAKE_THRESHOLD) {
            triggerWake();
        }
    } finally {
        isProcessing = false;
    }
}

// ----- UI trigger --------------------------------------------------------
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
        cycleMaxHistory = [];
    }, 2000);
}

// ----- audio capture ---------------------------------------------------
function setupAudioProcessing(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
    });

    deviceSampleRate = audioContext.sampleRate;
    console.log('[ANIQUEN] Requested', SAMPLE_RATE, 'Hz — actual AudioContext rate:', deviceSampleRate);

    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const scaled = new Float32Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            scaled[i] = inputData[i] * 32768;
        }

        const resampled = resampleTo16k(scaled, deviceSampleRate);

        audioBuffer.push(...resampled);
        if (audioBuffer.length > MAX_BUFFER_SAMPLES) {
            audioBuffer.splice(0, audioBuffer.length - MAX_BUFFER_SAMPLES);
        }

        if (!isWakeActive) {
            processAudioChunk().catch(e => console.warn(e));
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
                // turned OFF — browser AGC was behaving erratically on mobile;
                // we now do our own normalization in normalizeGain() instead.
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
