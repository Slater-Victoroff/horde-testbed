import { worldSetupConfig } from "./network.js";

export function setupScene(engine) {
    const scene = new BABYLON.Scene(engine);

    const groundSize = worldSetupConfig.ground_size
    
    // Setup camera
    const camera = new BABYLON.ArcRotateCamera("isoCam",
        BABYLON.Tools.ToRadians(-90), BABYLON.Tools.ToRadians(45), 25,
        new BABYLON.Vector3(0, 0, 0), scene
    );
    camera.attachControl(engine.getRenderingCanvas(), false);
    camera.inputs.clear();
    camera.lowerRadiusLimit = camera.upperRadiusLimit = camera.radius;
    camera.panningSensibility = 0;
    camera.wheelDeltaPercentage = 0;

    // Lighting
    const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
    light.intensity = 0.6;

    const directionalLight = new BABYLON.DirectionalLight("directionalLight", new BABYLON.Vector3(-1, -2, -1), scene);
    directionalLight.position = new BABYLON.Vector3(10, 10, 10); // Position the light
    directionalLight.intensity = 1.0;

    const shadowGenerator = new BABYLON.ShadowGenerator(1024, directionalLight);
    shadowGenerator.useBlurExponentialShadowMap = true; // Softer shadows
    shadowGenerator.blurKernel = 32;

    const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: groundSize, height: groundSize, subdivisions: 1 }, scene);
    ground.receiveShadows = true; // Enable shadows on the ground

    return { scene, camera, shadowGenerator };
}
