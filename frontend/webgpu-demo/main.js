import * as BABYLON from "https://cdn.babylonjs.com/babylon.js";

document.addEventListener("DOMContentLoaded", async () => {
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.WebGPUEngine(canvas);
    await engine.initAsync();
    console.log("WebGPU initialized");

    const scene = new BABYLON.Scene(engine);

    const camera = new BABYLON.ArcRotateCamera("camera", 0, 0, 1, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    camera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
    camera.orthoTop = 1;
    camera.orthoBottom = -1;
    camera.orthoLeft = -1;
    camera.orthoRight = 1;
    camera.setPosition(new BABYLON.Vector3(0, 0, -1));
    camera.fov = Math.PI / 2;

    const plane = BABYLON.MeshBuilder.CreatePlane("plane", { size: 2 }, scene);

    const shaderMaterial = new BABYLON.ShaderMaterial("spiralShader", scene, {
        vertex: "shader",
        fragment: "shader",
    }, {
        attributes: ["position", "uv"],
        uniforms: ["world", "worldView", "worldViewProjection", "time", "control"]
    });

    shaderMaterial.setFloat("time", 0);
    shaderMaterial.setFloats("control", [0, 0, 0, 0, 0, 0, 0, 0]);

    const texture = new BABYLON.Texture("/webgpu-demo/latent.exr", scene, false, false, BABYLON.Texture.NEAREST_SAMPLINGMODE);
    shaderMaterial.setTexture("latentTex", texture);

    plane.material = shaderMaterial;

    scene.registerBeforeRender(() => {
        const t = performance.now() / 1000;
        shaderMaterial.setFloat("time", t % 1); // normalized loop
        shaderMaterial.setFloats("control", [
            t % 1,
            Math.sin(t),
            Math.cos(t),
            Math.sin(2 * t),
            Math.cos(2 * t),
            Math.sin(3 * t),
            Math.cos(3 * t),
            0.0 // placeholder for style
        ]);
    });

    engine.runRenderLoop(() => {
        scene.render();
    });

    window.addEventListener("resize", () => {
        engine.resize();
    });
});
