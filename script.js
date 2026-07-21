// script_server.js – thin client: just streams mic audio to the server
// and reacts to detection results. No ONNX/ML runs on this device anymore.

const SAMPLE_RATE = 16000;

// CHANGE THIS to your server's address once deployed, e.g.:
// const SERVER_URL = 'wss://your-vm-domain-or-ip:8000/ws';
const SERVER_URL = 'wss://aniquen-wakeword.onrender.com/ws';

const statusEl = document.getElementById('status');
const bodyEl = document.body;

let audioContext = null;
let socket = null;
let isWakeActive = false;
let deviceSampleRate = SAMPLE_RATE;

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

let wakeTimeout = null;

function triggerWake() {
    if (isWakeActive) return;
    isWakeActive = true;
    bodyEl.classList.add('wake');
    statusEl.textContent = 'Wake Word Detected!';

    if (wakeTimeout) clearTimeout(wakeTimeout);
    wakeTimeout = setTimeout(() => {
        bodyEl.classList.remove('wake');
        statusEl.textContent = 'Listening…';
        isWakeActive = false;
        wakeTimeout = null;
    }, 2000);
}

function connectSocket() {
    socket = new WebSocket(SERVER_URL);
    socket.binaryType = 'arraybuffer';

    socket.onopen = () => {
        console.log('[ANIQUEN] Connected to server.');
        statusEl.textContent = 'Listening…';
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.idle) {
            // server skipped this cycle (VAD gate) — nothing to do
            return;
        }
        console.log(
            '[ANIQUEN] rms:', data.rms,
            ' gain:', data.gain,
            ' prob:', data.prob,
            ' rolling max:', data.rolling_max
        );
        if (data.detected) {
            triggerWake();
        }
    };

    socket.onclose = () => {
        console.warn('[ANIQUEN] Server connection closed — retrying in 2s...');
        statusEl.textContent = 'Reconnecting…';
        setTimeout(connectSocket, 2000);
    };

    socket.onerror = (err) => {
        console.error('[ANIQUEN] Socket error:', err);
    };
}

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

        // send raw float32 PCM bytes straight to the server
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(resampled.buffer);
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
}

async function init() {
    try {
        connectSocket();
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: SAMPLE_RATE,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: true,
            }
        });
        setupAudioProcessing(stream);
        console.log('[ANIQUEN] Microphone ready, streaming to server.');
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
    if (audioContext) audioContext.close().catch(() => {});
    if (socket) socket.close();
});
