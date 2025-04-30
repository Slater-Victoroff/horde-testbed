export async function createNeuralShaderMaterial(scene, kernel) {
    try {
        BABYLON.Engine.ShadersRepository = ""; // Prevent Babylon.js from looking for .fx files
        BABYLON.Effect.LogShaderCodeOnCompilation = true; // Log shader code during compilation
        // Load the vertex shader
        delete BABYLON.Effect.ShadersStore["neuralVertexShader"];
        BABYLON.Effect.ShadersStore["neural2VertexShader"] = `
            precision highp float;
            attribute vec3 position;
            attribute vec2 uv;
            uniform mat4 worldViewProjection;
            out vec2 vUv;
            void main(void) {
                gl_Position = worldViewProjection * vec4(position, 1.0);
                vUv = uv;
            }
        `;

        // Load the fragment shader
        const shaderResponse = await fetch('http://3.226.12.50:8000/assets/neural_shader/shader.frag?nocache=${Date.now()}');
        if (!shaderResponse.ok) {
            throw new Error(`Failed to load fragment shader: ${shaderResponse.statusText}`);
        }
        const fragmentShaderCode = await shaderResponse.text();
        delete BABYLON.Effect.ShadersStore["neuralFragmentShader"];
        BABYLON.Effect.ShadersStore["neural2FragmentShader"] = fragmentShaderCode;

        const shaderMaterial = new BABYLON.ShaderMaterial("neuralShader", scene, {
            vertex: "neural2",
            fragment: "neural2",
        }, {
            attributes: ["position", "uv"],
            uniforms: ["debug_layer", "debug_uv", "worldViewProjection", "control"],
        });


        BABYLON.ExrLoaderGlobalConfiguration.DefaultOutputType = BABYLON.EXROutputType.Float;
        // Set resolution and grayscale texture
        const latentText = new BABYLON.Texture(
            `http://3.226.12.50:8000/assets/${kernel.material}/latent_y.exr`,
            scene,
            false,
            false,
            BABYLON.Texture.NEAREST_SAMPLINGMODE,
            null,
            null,
            BABYLON.Engine.TEXTURETYPE_FLOAT
        );
        shaderMaterial.setTexture("latentText", latentText);

        const weightText = new BABYLON.Texture(
            `http://3.226.12.50:8000/assets/${kernel.material}/weights.exr`,
            scene,
            false,
            false,
            BABYLON.Texture.NEAREST_SAMPLINGMODE,
            null,
            null,
            BABYLON.Engine.TEXTURETYPE_FLOAT
        );
        shaderMaterial.setTexture("weightText", weightText);
        latentText.onLoadObservable.add(() => {
            console.log("Latent texture loaded:", latentText);
        });
        weightText.onLoadObservable.add(() => {
            console.log("Weight texture loaded:", weightText);
        });
        const controlVector = [0.0, 0.0, 1.0, 0.0];
        shaderMaterial.setFloats("control", controlVector);
        shaderMaterial.setInt("debug_layer", 3);
        shaderMaterial.setFloats("debug_uv", [0.5, 0.5]);
        return shaderMaterial;
    } catch (error) {
        console.error("Error creating neural shader material:", error);
        return null;
    }
}
