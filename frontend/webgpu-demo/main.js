import { computeTrunkAt, computeTargetedEncodings, buildLayerMap, getWeight } from "./cpu_inference.js";

const canvas = document.getElementById("webgpu-canvas");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();

canvas.width = canvas.clientWidth * window.devicePixelRatio;
canvas.height = canvas.clientHeight * window.devicePixelRatio;

context.configure({
    device,
    format,
    alphaMode: "opaque",
    size: [canvas.width, canvas.height],
});

const shaderModule = await fetch(`shader.wgsl?cacheBust=${Date.now()}`)
    .then(r => r.text())
    .then(code => device.createShaderModule({ code }));

const vertices = new Float32Array([
    -1, -1, 0, 1,
     1, -1, 1, 1,
    -1,  1, 0, 0,
     1,  1, 1, 0,
]);

// console.log("Wah! look at me!");
// console.log(computeTargetedEncodings(
//     [0.4, 0.3],
//     16,
//     { scheme: "spiral", norm2Pi: true, includeNorm: true }
// ));

const debugCount = 8;
const debugBuffer = device.createBuffer({
    size: debugCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
});

const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const controlData = new Float32Array([0.0, 0.0, 0.0, 0.0]);

const controlBuffer = device.createBuffer({
    size: controlData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
// Load combined model file and manifest
const [manifest, rawBuffer] = await Promise.all([
    fetch(`model_manifest.json?cacheBust=${Date.now()}`).then(r => r.json()),
    fetch(`model_weights.bin?cacheBust=${Date.now()}`).then(r => r.arrayBuffer())
]);

const floatBuffer = new Float32Array(rawBuffer);

const layerMap = buildLayerMap(manifest.layers);

console.log(
  "JS decoder.layers.0.weight[0,0] =",
  getWeight(layerMap, floatBuffer, "decoder.layers.0.weight", 0, 0)
);

console.log(
  "JS decoder.layers.0.bias[0]     =",
  getWeight(layerMap, floatBuffer, "decoder.layers.0.bias", 0)
);

console.log(
  "JS decoder.film.bias[10]        =",
  getWeight(layerMap, floatBuffer, "decoder.film.bias", 10)
);
console.log(
    "JS time_embed  [11,4]  =",
    getWeight(layerMap, floatBuffer, "decoder.time_embed.0.weight", 11, 4)
);
  
console.log(
    "JS film.weight [90,33] =",
    getWeight(layerMap, floatBuffer, "decoder.film.weight", 90, 33)
);

console.log(
    "JS L2.weight  [17,42] =",
    getWeight(layerMap, floatBuffer, "decoder.layers.2.weight", 17, 42)
);

// Extract the shared_latent entry
const latentMeta = manifest.layers.find(l => l.name === "shared_latent");
if (!latentMeta) throw new Error("shared_latent not found in manifest");

const latentOffset = latentMeta.offset / 4; // offset is in bytes, so divide by 4 for float32
const latentLength = (latentMeta.size || (latentMeta.shape[0] * latentMeta.shape[1] * latentMeta.shape[2]));
const latentData = floatBuffer.subarray(latentOffset, latentOffset + latentLength);

const latentWidth = latentMeta.shape[1]; // 128
const latentHeight = latentMeta.shape[0]; // 240

const numLayers = manifest.layers.length - 1;
const modelLayers = manifest.layers.filter(layer => layer.name !== "shared_latent");
const modelStart = Math.min(...modelLayers.map(l => l.offset / 4));
const modelEnd   = Math.max(...modelLayers.map(l => l.offset / 4 + l.size));

const offsetData = new Uint32Array(numLayers * 2);
for (let i = 0; i < modelLayers.length; i++) {
    const lay = modelLayers[i];
    // weight comes first in the manifest for each layer
    offsetData[i*2 + 0] = lay.offset   / 4;           // weightBase
    offsetData[i*2 + 1] = (lay.offset + lay.size*4) / 4; // biasBase
}

const offsetBuf = device.createBuffer({
    size: offsetData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(offsetBuf, 0, offsetData);

console.log('modelStart', modelStart);
console.log('modelEnd', modelEnd);

const metaData = new Uint32Array([modelStart]);   // 4-byte uint is fine
const metaBuf  = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(metaBuf, 0, metaData);

const modelLength = modelEnd - modelStart;
const modelData = floatBuffer.subarray(modelStart, modelEnd);

const modelTexels = Math.ceil(modelLength);
const modelTexWidth = 256;
const modelTexHeight = Math.ceil(modelTexels / modelTexWidth);
console.log('modelTexWidth', modelTexWidth);
console.log('modelTexHeight', modelTexHeight);
const paddedModelData = new Float32Array(modelTexWidth * modelTexHeight);
paddedModelData.set(modelData);
const modelBuffer = device.createBuffer({
    size: paddedModelData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  
device.queue.writeBuffer(modelBuffer, 0, paddedModelData);

console.log('last float sent to GPU', paddedModelData[paddedModelData.length - 1]);
console.log('first float of last row', paddedModelData[paddedModelData.length - modelTexWidth]);

const latentTex = device.createTexture({
    size: [latentWidth, latentHeight],
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

device.queue.writeTexture(
    { texture: latentTex },
        latentData,
    {
        offset: 0,
        bytesPerRow: latentWidth * 4 * 4, // 4 bytes per float
        rowsPerImage: latentHeight,
    },
    {
        width: latentWidth,
        height: latentHeight,
        depthOrArrayLayers: 1,
    }
);

const latentSampler = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
});

const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "non-filtering" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ],
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
});

const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [{
            arrayStride: 4 * 4,
            attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x2" },
            ],
        }],
    },
    fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format }],
    },
    primitive: { topology: "triangle-strip" },
});

const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: controlBuffer, } },
        { binding: 1, resource: latentTex.createView() },
        { binding: 2, resource: latentSampler },
        { binding: 3, resource: { buffer: modelBuffer } },
        { binding: 4, resource: { buffer: debugBuffer } },
        { binding: 5, resource: { buffer: metaBuf } },
        { binding: 6, resource: { buffer: offsetBuf } },
    ],
});

let mousePosition = { x: 0, y: 0 };
canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mousePosition.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mousePosition.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
});

const idleBase    = 0.42;
const roamSpeed     = 0.02;
const flickerAmp  = 0.75;
const flickerInterval = 0.8;  
let startTime = performance.now();

let prevFlickTarget = 0;
let nextFlickTarget = (Math.random()*2 - 1) * flickerAmp;
let flickerTime0    = 0;

async function frame() {
    const currentTime = performance.now();
    const elapsedTime = (currentTime - startTime) / 1000; // Convert to seconds
    const circleTime = (elapsedTime % 10.0) / 10.0;

    const baseT = (idleBase + elapsedTime * roamSpeed) % 1.0;
    if (elapsedTime - flickerTime0 >= flickerInterval) {
        flickerTime0    += flickerInterval;
        prevFlickTarget  = nextFlickTarget;
        nextFlickTarget  = (Math.random()*2 - 1) * flickerAmp;
    }
    const phase      = ((elapsedTime - flickerTime0) / flickerInterval);
    const flickerVal = prevFlickTarget * (1-phase) + nextFlickTarget * phase;
      
    controlData[0] = circleTime;
    controlData[1] = flickerVal;
    controlData[2] = 0.0;
    controlData[3] = 0.0;
    device.queue.writeBuffer(controlBuffer, 0, controlData);

    const sizeX = latentWidth  * 2.0 / canvas.width;
    const sizeY = latentHeight * 2.0 / canvas.height;
    
    // ──   shift the quad up by half its height   ─────────────────────────────────
    // anchorY = +½ * sizeY  → mouse sits on the bottom edge
    const quadCenterY = mousePosition.y + (sizeY / 2.5);
    const quadCenterX = mousePosition.x;          // still centred in X
    
    // ──   build the vertex array (v coordinates already flipped)   ───────────────
    const vertices = new Float32Array([
      quadCenterX - sizeX / 2,  quadCenterY - sizeY / 2,   0, 1,   // bottom-left
      quadCenterX + sizeX / 2,  quadCenterY - sizeY / 2,   1, 1,   // bottom-right
      quadCenterX - sizeX / 2,  quadCenterY + sizeY / 2,   0, 0,   // top-left
      quadCenterX + sizeX / 2,  quadCenterY + sizeY / 2,   1, 0,   // top-right
    ]);

    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
    });

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4, 1, 0, 0);
    pass.end();    
    device.queue.submit([encoder.finish()]);

    const readback = device.createBuffer({
        size: debugCount * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const copyEnc = device.createCommandEncoder();
    copyEnc.copyBufferToBuffer(debugBuffer, 0, readback, 0, debugCount * 4);
    device.queue.submit([copyEnc.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readback.getMappedRange());
    // console.log("WGSL debug_data:", Array.from(data));
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);