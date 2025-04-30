precision highp float;
uniform sampler2D latentTex;
uniform sampler2D weightsTex;
uniform vec4 control;
uniform float layer_offsets[5];
uniform int debug_layer;
uniform vec2 debug_uv;

in vec2 vUv;
out vec4 fragColor;

float gelu(float x) {{
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159265359) * (x + 0.044715 * x * x * x)));
}}

float sigmoid(float x) {{
    return 1.0 / (1.0 + exp(-x));
}}

float get_weight(int layer, int i, int j, int input_size, int output_size, float tex_width) {{
    // Calculate the UV coordinates for the weight
    float text_width = float(tex_width);
    float depth = float(i * input_size + j) / 4.0;
    float index = float(layer_offsets[layer]) + depth;

    float x = (mod(index, tex_width)) / tex_width;
    float y = floor(index / tex_width) / tex_width;

    // Determine the channel to use based on the index
    int channel = int(mod(index * 4.0, 4.0));  // Adjusted for proper channel selection
    if (channel == 0) {{
        return texture(weightsTex, vec2(x, y)).r;
    }} else if (channel == 1) {{
        return texture(weightsTex, vec2(x, y)).g;
    }} else if (channel == 2) {{
        return texture(weightsTex, vec2(x, y)).b;
    }} else {{
        return texture(weightsTex, vec2(x, y)).a;
    }}
}}

float get_bias(int layer, int i, int output_size, float tex_width) {{
    // Calculate the UV coordinates for the bias
    float index = float(layer_offsets[layer + 1]) - float(output_size) / 4.0 + float(i) / 4.0;
    float x = mod(index, tex_width) / tex_width;
    float y = floor(index / tex_width) / tex_width;

    // Determine the channel to use based on the index
    int channel = int(mod(index * 4.0, 4.0));  // Adjusted for proper channel selection
    if (channel == 0) {{
        return texture(weightsTex, vec2(x, y)).r;
    }} else if (channel == 1) {{
        return texture(weightsTex, vec2(x, y)).g;
    }} else if (channel == 2) {{
        return texture(weightsTex, vec2(x, y)).b;
    }} else {{
        return texture(weightsTex, vec2(x, y)).a;
    }}
}}

void main() {
    float grayscale = texture(latentTex, vUv).r;

    // Initialize the input to the first layer
    float inputVec[7];
    inputVec[0] = grayscale;      
    inputVec[1] = vUv.x;            // Positional encoding
    inputVec[2] = vUv.y;            
    inputVec[3] = float(control.x);       // Control vector
    inputVec[4] = float(control.y);       
    inputVec[5] = float(control.z);       
    inputVec[6] = float(control.w);

    float hidden_0[7];
    for (int i = 0; i < 7; i++) {
        hidden_0[i] = inputVec[i];
    }

    float prehidden_1[32];
    float rehidden_1[32];
    float hidden_1[32];
    for (int i = 0; i < 32; i++) {
        prehidden_1[i] = get_bias(0, i, 32, 34.0);
        rehidden_1[i] = prehidden_1[i];
        for (int j = 0; j < 7; j++) {
            rehidden_1[i] += get_weight(0, i, j, 7, 32, 34.0) * hidden_0[j];
        }
    }

    for (int i = 0; i < 32; i++) {
        hidden_1[i] = gelu(rehidden_1[i]);
    }

    float prehidden_2[64];
    float rehidden_2[64];
    float hidden_2[64];
    for (int i = 0; i < 64; i++) {
        prehidden_2[i] = get_bias(1, i, 64, 34.0);
        rehidden_2[i] = prehidden_2[i];
        for (int j = 0; j < 32; j++) {
            rehidden_2[i] += get_weight(1, i, j, 32, 64, 34.0) * hidden_1[j];
        }
    }

    for (int i = 0; i < 64; i++) {
        hidden_2[i] = gelu(rehidden_2[i]);
    }

    float prehidden_3[32];
    float rehidden_3[32];
    float hidden_3[32];
    for (int i = 0; i < 32; i++) {
        prehidden_3[i] = get_bias(2, i, 32, 34.0);
        rehidden_3[i] = prehidden_3[i];
        for (int j = 0; j < 64; j++) {
            rehidden_3[i] += get_weight(2, i, j, 64, 32, 34.0) * hidden_2[j];
        }
    }

    for (int i = 0; i < 32; i++) {
        hidden_3[i] = gelu(rehidden_3[i]);
    }

    float prehidden_4[3];
    float rehidden_4[3];
    float hidden_4[3];
    for (int i = 0; i < 3; i++) {
        prehidden_4[i] = get_bias(3, i, 3, 34.0);
        rehidden_4[i] = prehidden_4[i];
        for (int j = 0; j < 32; j++) {
            rehidden_4[i] += get_weight(3, i, j, 32, 3, 34.0) * hidden_3[j];
        }
    }

    for (int i = 0; i < 3; i++) {
        hidden_4[i] = sigmoid(rehidden_4[i]);
    }

    if (debug_layer == 1 && abs(vUv.x - debug_uv.x) < 0.001 && abs(vUv.y - debug_uv.y) < 0.001) {
        fragColor = vec4(1.23, texture(latentTex, vec2(0.0, 1.0)).r, texture(latentTex, vec2(1.0, 0.0)).r, 1.0);
        return;
    }

    if (debug_layer == 2 && abs(vUv.x - debug_uv.x) < 0.001 && abs(vUv.y - debug_uv.y) < 0.001) {
        fragColor = vec4(1.23, texture(latentTex, vec2(0.0, 1.0)).r, texture(latentTex, vec2(1.0, 0.0)).r, 1.0);
        return;
    }

    if (debug_layer == 3 && abs(vUv.x - debug_uv.x) < 0.001 && abs(vUv.y - debug_uv.y) < 0.001) {
        fragColor = vec4(1.23, texture(latentTex, vec2(0.0, 1.0)).r, texture(latentTex, vec2(1.0, 0.0)).r, 1.0);
        return;
    }

    if (debug_layer == 4 && abs(vUv.x - debug_uv.x) < 0.001 && abs(vUv.y - debug_uv.y) < 0.001) {
        fragColor = vec4(1.23, texture(latentTex, vec2(0.0, 1.0)).r, texture(latentTex, vec2(1.0, 0.0)).r, 1.0);
        return;
    }

    float debug_effect = float(debug_layer) * debug_uv.x * 0.00001;
    fragColor += vec4(debug_effect, debug_effect, debug_effect, 0.0);
}
