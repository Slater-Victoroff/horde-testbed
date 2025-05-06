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
    -1, -1, 0, 0,
     1, -1, 1, 0,
    -1,  1, 0, 1,
     1,  1, 1, 1,
]);

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

const latentData = await fetch(`latent.bin?cacheBust=${Date.now()}`)
    .then(response => response.arrayBuffer())
    .then(buffer => new Float32Array(buffer));

const latentWidth = 128;
const latentHeight = 240;
const expectedSize = latentWidth * latentHeight * 4; // 4 channels (RGBA)
if (latentData.length !== expectedSize) {
    throw new Error(`Expected ${expectedSize} floats, but got ${latentData.length}`);
}

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
    ],
});

let mousePosition = { x: 0, y: 0 };
canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mousePosition.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mousePosition.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
});

let startTime = performance.now();

function frame() {
    const currentTime = performance.now();
    const elapsedTime = (currentTime - startTime) / 1000; // Convert to seconds
    const loopedTime = (elapsedTime % 3) / 3;
    controlData[0] = loopedTime;
    controlData[1] = 0.0;
    controlData[2] = 0.0;
    controlData[3] = 0.0;
    device.queue.writeBuffer(controlBuffer, 0, controlData);

    const pxWidth = 128;
    const pxHeight = 240;
    const sizeX = pxWidth * 2.0 / canvas.width;
    const sizeY = pxHeight * 2.0 / canvas.height;


    const vertices = new Float32Array([
        mousePosition.x - sizeX / 2, mousePosition.y - sizeY / 2, 0, 0,
        mousePosition.x + sizeX / 2, mousePosition.y - sizeY / 2, 1, 0,
        mousePosition.x - sizeX / 2, mousePosition.y + sizeY / 2, 0, 1,
        mousePosition.x + sizeX / 2, mousePosition.y + sizeY / 2, 1, 1,
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
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);