import os
import json
import textwrap
import torch
import moderngl
import numpy as np
from PIL import Image

import torch.nn as nn
import OpenEXR
import Imath

# Vertex shader (fullscreen quad)
VERTEX_SHADER = """
layout(location = 0) in vec2 in_vert;
out vec2 vUv;
void main() {
    vUv = in_vert * 0.5 + 0.5;  // Map clip space to UV space [0, 1]
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

def decoder_to_glsl(decoder, weights_path, debug_mode=False):
    """
    Made to target GLSL version 300 es because that is what Babylon supports.
    """

    glsl_code = textwrap.dedent(f"""
        precision highp float;
        uniform sampler2D latentTex;
        uniform sampler2D weightsTex;
        uniform vec4 control;
        uniform float layer_offsets[{int((len(decoder.layers) / 2) + 1)}];
        """
        ).strip()
    if debug_mode:
        glsl_code += textwrap.dedent("""
            uniform int debug_layer;
            uniform vec2 debug_uv;
            """
            )
    glsl_code += textwrap.dedent("""
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
        """
        )

    exr_file = OpenEXR.InputFile(weights_path)
    dw = exr_file.header()["dataWindow"]
    weights_tex_width = int(dw.max.x - dw.min.x + 1)  # Assuming square texture

    print("weights_tex_width", weights_tex_width)
    weights_data = np.frombuffer(
        exr_file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)) +
        exr_file.channel("G", Imath.PixelType(Imath.PixelType.FLOAT)) +
        exr_file.channel("B", Imath.PixelType(Imath.PixelType.FLOAT)) +
        exr_file.channel("A", Imath.PixelType(Imath.PixelType.FLOAT)),
        dtype=np.float32
    )

    input_size = 1 + 2 + decoder.num_images  # Grayscale + positional encodings + control

    # Initialize the main function
    glsl_code += textwrap.dedent(f"""
        void main() {{
            float grayscale = texture(latentTex, vUv).r;

            // Initialize the input to the first layer
            float inputVec[{input_size}];
            inputVec[0] = grayscale;      
            inputVec[1] = vUv.x;            // Positional encoding
            inputVec[2] = vUv.y;            
            inputVec[3] = float(control.x);       // Control vector
            inputVec[4] = float(control.y);       
            inputVec[5] = float(control.z);       
            inputVec[6] = float(control.w);

            float hidden_{0}[{input_size}];
            for (int i = 0; i < {input_size}; i++) {{
                hidden_{0}[i] = inputVec[i];
            }}
        """
        )

    current_size = input_size
    last_hidden = 0
    for layer in decoder.layers:
        if isinstance(layer, nn.Linear):
            output_size = layer.out_features
            glsl_code += textwrap.indent(textwrap.dedent(f"""
                float prehidden_{last_hidden + 1}[{output_size}];
                float rehidden_{last_hidden + 1}[{output_size}];
                float hidden_{last_hidden + 1}[{output_size}];
                for (int i = 0; i < {output_size}; i++) {{
                    prehidden_{last_hidden + 1}[i] = get_bias({last_hidden}, i, {output_size}, {weights_tex_width}.0);
                    rehidden_{last_hidden + 1}[i] = prehidden_{last_hidden + 1}[i];
                    for (int j = 0; j < {current_size}; j++) {{
                        rehidden_{last_hidden + 1}[i] += get_weight({last_hidden}, i, j, {current_size}, {output_size}, {weights_tex_width}.0) * hidden_{last_hidden}[j];
                    }}
                }}
            """
            ), "    ")
            current_size = output_size
            last_hidden = last_hidden + 1
        elif isinstance(layer, nn.GELU):
            glsl_code += textwrap.indent(textwrap.dedent(f"""
                for (int i = 0; i < {current_size}; i++) {{
                    hidden_{last_hidden}[i] = gelu(rehidden_{last_hidden}[i]);
                }}
                """
            ), "    ")
        elif isinstance(layer, nn.Sigmoid):
            glsl_code += textwrap.indent(textwrap.dedent(f"""
                for (int i = 0; i < {current_size}; i++) {{
                    hidden_{last_hidden}[i] = sigmoid(rehidden_{last_hidden}[i]);
                }}
                """
            ), "    ")

    if debug_mode:
        for layer in range(1, last_hidden + 1):
            glsl_code += textwrap.indent(textwrap.dedent(f"""
                if (debug_layer == {layer} && abs(vUv.x - debug_uv.x) < 0.001 && abs(vUv.y - debug_uv.y) < 0.001) {{
                    fragColor = vec4(1.23, texture(latentTex, vec2(0.0, 1.0)).r, texture(latentTex, vec2(1.0, 0.0)).r, 1.0);
                    return;
                }}
            """), "    ")
        glsl_code += textwrap.dedent(f"""
            float debug_effect = float(debug_layer) * debug_uv.x * 0.00001;
            fragColor += vec4(debug_effect, debug_effect, debug_effect, 0.0);
        }}
        """)
    else:
        glsl_code += textwrap.dedent(f"""
            fragColor = vec4(hidden_{last_hidden}[0], hidden_{last_hidden}[1], hidden_{last_hidden}[2], 1.0);
        }}
        """
        )
    return glsl_code


def save_weights_to_exr(decoder, path="weights.exr"):
    # TODO: Simplify so that r are weights, b bias, g weights, a bias
    weights = []
    layer_offsets = [0]
    for layer in decoder.layers:
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().cpu().numpy().astype(np.float32).flatten()
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            weights.extend(w)
            print("weights", w[:10])
            weights.extend(b)
            print("bias", b[:32])
            if len(weights) % 4 != 0:
                print("Padding weights to 4-byte alignment")
                print("weights length", len(weights))
                print("weights mod 4", len(weights) % 4)
                weights.extend([0.0] * (4 - len(weights) % 4))
            layer_offsets.append(len(weights) / 4)

    rgba = np.array(weights, dtype=np.float32).reshape(-1, 4)

    # Pad to square
    size = int(np.ceil(np.sqrt(len(rgba))))
    pad = size**2 - len(rgba)
    if pad > 0:
        rgba = np.vstack([rgba, np.zeros((pad, 4), dtype=np.float32)])

    rgba = rgba.reshape((size, size, 4))

    # Save as 32-bit float RGBA EXR
    header = OpenEXR.Header(size, size)
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "A": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }
    exr = OpenEXR.OutputFile(path, header)

    # Split RGBA into separate channels
    r = rgba[:, :, 0].flatten().tobytes()
    g = rgba[:, :, 1].flatten().tobytes()
    b = rgba[:, :, 2].flatten().tobytes()
    a = rgba[:, :, 3].flatten().tobytes()

    print("Red channel start: ", rgba[:, :, 0].flatten()[:10])
    print("Green channel start: ", rgba[:, :, 1].flatten()[:10])
    print("Blue channel start: ", rgba[:, :, 2].flatten()[:10])
    print("Alpha channel start: ", rgba[:, :, 3].flatten()[:10])

    # Write channels to EXR
    exr.writePixels({"R": r, "G": g, "B": b, "A": a})
    exr.close()
    return layer_offsets


def save_latent_to_exr(latent, path="latent.exr"):
    latent_np = latent.cpu().numpy().astype(np.float32)
    print("latent_np", latent_np.shape)
    height, width, _ = latent_np.shape
    channels = latent_np.shape[2] if len(latent_np.shape) > 2 else 1
    if channels != 4:
        raise ValueError("Latent tensor must have 4 channels (RGBA).")

    # Prepare EXR file
    header = OpenEXR.Header(width, height)
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "A": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }
    exr = OpenEXR.OutputFile(path, header)

    # Split RGBA channels
    r = latent_np[:, :, 0].flatten().tobytes()
    g = latent_np[:, :, 1].flatten().tobytes()
    b = latent_np[:, :, 2].flatten().tobytes()
    a = latent_np[:, :, 3].flatten().tobytes()

    # Write RGBA data
    exr.writePixels({"R": r, "G": g, "B": b, "A": a})
    exr.close()


def flatten_and_pad(array, target_size):
    """
    Flatten a 2D array and pad it to the target size.
    """
    flat_array = array.flatten()
    if len(flat_array) < target_size:
        flat_array = np.pad(flat_array, (0, target_size - len(flat_array)), mode="constant")
    return flat_array


def compare_decoder_and_shader(decoder, glsl_path, grayscale_path, weights_path, offsets_path, num_images=4, debug_layer=1):
    glsl_code = load_glsl_code(glsl_path)
    latent_texture, latent_texture_tensor, width, height = load_latent_texture(grayscale_path)
    print("latent_texture", latent_texture.shape)
    print("latent_texture_start", latent_texture[0, :10])
    weights_texture, weights_tex_width = load_weights_texture(weights_path)
    control_vectors = torch.eye(num_images, dtype=torch.float32)
    offsets = json.load(open(offsets_path, "r"))
    layer_offsets = [int(offset) for offset in offsets["layer_offsets"]]

    pytorch_outputs, glsl_outputs = [], []
    ctx, program, vao, fbo = setup_opengl_context(glsl_code, latent_texture, weights_texture, width, height, weights_tex_width)

    debug_x, debug_y = 0, 0
    debug_uv = (debug_x / width, debug_y / height)  # Convert to UV coordinates
    if "debug_uv" in program:
        program["debug_uv"].value = debug_uv
        print("#" * 20)
        print(f"Debug UV: {debug_uv}")
        print(f"Debug pixel: ({debug_x}, {debug_y})")
    if "debug_layer" in program:
        program["debug_layer"].value = 1
        print("$" * 20)

    for control_vector in control_vectors:
        glsl_output = render_with_shader(ctx, program, vao, fbo, control_vector, layer_offsets, width, height)
        
        glsl_pixel_output = glsl_output[debug_y, debug_x, :3]  # RGB only

        # Get the PyTorch hidden layer output for the debug pixel
        grayscale_value = latent_texture_tensor[debug_y, debug_x].item()
        pytorch_hidden_output = decoder.decode_single_pixel(
            grayscale_value,
            debug_uv[0],
            debug_uv[1],
            control_vector,
            device="cuda",
            return_hidden_layer=0
        ).cpu().detach().numpy()

        diff = np.abs(pytorch_hidden_output[:, :3].T - glsl_pixel_output)
        print(f"Control Vector: {control_vector}")
        print(f"Layer {2}:")
        print(f"  GLSL Output: {glsl_pixel_output}")
        print(f"  PyTorch Output: {pytorch_hidden_output}")
        print(f"  Difference: {diff}")

        print(f"GLSL output shape: {glsl_output.shape}")
        print(f"GLSL output min: {np.min(glsl_output)}, max: {np.max(glsl_output)}")
        print(f"GLSL output mean: {np.mean(glsl_output)}, std: {np.std(glsl_output)}")
        glsl_outputs.append(glsl_output)

        pytorch_output = decoder.decode_full_image(latent_texture_tensor, control_vector).cpu().detach().numpy()
        pytorch_outputs.append(pytorch_output)

    compare_outputs(pytorch_outputs, glsl_outputs)


def load_glsl_code(glsl_path):
    with open(glsl_path, "r") as f:
        return f.read()


def load_latent_texture(grayscale_path):
    exr_file = OpenEXR.InputFile(grayscale_path)
    dw = exr_file.header()["dataWindow"]
    width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    grayscale_data = np.frombuffer(exr_file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
    latent_texture = grayscale_data.reshape(height, width)
    latent_texture_tensor = torch.from_numpy(grayscale_data.copy()).reshape(height, width).to(dtype=torch.float32, device="cuda")
    return latent_texture, latent_texture_tensor, width, height


def load_weights_texture(weights_path):
    exr_file = OpenEXR.InputFile(weights_path)
    weights_tex_width = int(exr_file.header()["dataWindow"].max.x - exr_file.header()["dataWindow"].min.x + 1)

    # Read each channel separately
    r_channel = np.frombuffer(exr_file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
    g_channel = np.frombuffer(exr_file.channel("G", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
    b_channel = np.frombuffer(exr_file.channel("B", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
    a_channel = np.frombuffer(exr_file.channel("A", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)

    # Interleave the channels to reconstruct the original RGBA order
    weights_data = np.empty((r_channel.size * 4,), dtype=np.float32)
    weights_data[0::4] = r_channel
    weights_data[1::4] = g_channel
    weights_data[2::4] = b_channel
    weights_data[3::4] = a_channel

    # Reshape into a 2D RGBA texture
    weights_texture = weights_data.reshape(weights_tex_width, weights_tex_width, 4)
    return weights_texture, weights_tex_width


def setup_opengl_context(glsl_code, latent_texture, weights_texture, width, height, weights_tex_width):
    ctx = moderngl.create_context(require=300, standalone=True, backend="egl")
    vertex_shader = "#version 300 es\n" + VERTEX_SHADER
    fragment_shader = "#version 300 es\n" + glsl_code
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    for name in program:
        member = program[name]
        print(name, type(member), member)
    vertices = np.array([
        -1.0, -1.0,  # Bottom-left
         1.0, -1.0,  # Bottom-right
        -1.0,  1.0,  # Top-left
         1.0,  1.0,  # Top-right
    ], dtype="f4")
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(program, vbo, "in_vert")

    latent_texture_gl = ctx.texture((width, height), 1, latent_texture.tobytes(), dtype="f4")
    latent_texture_gl.use(location=0)
    program["latentTex"].value = 0

    print("weights_texture", weights_texture.shape)
    print("weights_tex_width", weights_tex_width)
    print("weights_texture_start", weights_texture[0, :10, 0])
    print("weights_texture_start", weights_texture[:10, 0, 0])
    print("weights_texture_start", weights_texture[0, 0, :])
    weights_texture_gl = ctx.texture((weights_tex_width, weights_tex_width), 4, weights_texture.tobytes(), alignment=4, dtype="f4")
    weights_texture_gl.filter = (moderngl.NEAREST, moderngl.NEAREST)

    print("weights_texture_gl_start", weights_texture_gl.size)
    print("weights_texture_gl_start", type(weights_texture_gl.read()))
    print("weights_texture_gl.read() sample:", np.frombuffer(weights_texture_gl.read(), dtype=np.float32)[:10])
    weights_texture_gl.use(location=1)
    program["weightsTex"].value = 1

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4, dtype="f4")])
    fbo.use()

    return ctx, program, vao, fbo


def render_with_shader(ctx, program, vao, fbo, control_vector, layer_offsets, width, height):
    if "control" in program:
        program["control"].value = tuple(control_vector)
    if "layer_offsets" in program:
        program["layer_offsets"].write(np.array(layer_offsets, dtype=np.float32))

    fbo.clear()
    vao.render(moderngl.TRIANGLE_STRIP)

    data = fbo.read(components=4, dtype="f4")
    return np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)


def compare_outputs(pytorch_outputs, glsl_outputs):
    for i, (pytorch_output, glsl_output) in enumerate(zip(pytorch_outputs, glsl_outputs)):
        # Remove the alpha channel from the GLSL output
        glsl_output_rgb = glsl_output[:, :, :3]  # Keep only the RGB channels

        # Compute the difference
        diff = np.abs(pytorch_output - glsl_output_rgb)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        max_diff_location = np.unravel_index(np.argmax(diff), diff.shape)

        print(f"Control Vector {i}:")
        print(f"  Max difference = {max_diff:.6f} at {max_diff_location}")
        print(f"  Mean difference = {mean_diff:.6f}")
        print(f"  Std deviation = {std_diff:.6f}")

        if max_diff > 1e-3:  # Tolerance threshold
            print(f"  Warning: Significant difference detected for Control Vector {i}!")

        # Save PyTorch output as an image
        pytorch_image = (pytorch_output * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pytorch_image).save(f"pytorch_output_{i}.png")

        # Save GLSL output as an image
        glsl_image = (glsl_output_rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(glsl_image).save(f"glsl_output_{i}.png")

        # Save the difference as an image
        diff_image = (diff / max_diff * 255).clip(0, 255).astype(np.uint8)  # Normalize to [0, 255]
        Image.fromarray(diff_image).save(f"diff_output_{i}.png")

    print("Comparison complete. Outputs and differences saved as images.")
