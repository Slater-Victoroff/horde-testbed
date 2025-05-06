@group(0) @binding(0) var<uniform> control: vec4<f32>;
@group(0) @binding(1) var latentTex: texture_2d<f32>;
@group(0) @binding(2) var latentSampler: sampler;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

fn spiral_embed(n: i32, x: f32) -> array<f32, 2 * 3> {
  var out: array<f32, 6>;
  for (var i = 1; i <= 3; i = i + 1) {
    let idx = (i - 1) * 2;
    out[idx] = sin(f32(i) * x);
    out[idx + 1] = cos(f32(i) * x);
  }
  return out;
}

fn get_spiral_pos(uv: vec2<f32>) -> array<f32, 12> {
  let x = uv.x * 2.0 * 3.141592;
  let y = uv.y * 2.0 * 3.141592;
  let sx = spiral_embed(3, x); // [sin(x), cos(x), sin(2x), ...]
  let sy = spiral_embed(3, y);
  var out: array<f32, 12>;
  for (var i = 0; i < 6; i = i + 1) {
    out[i] = sx[i];
    out[i + 6] = sy[i];
  }
  return out;
}

fn get_spiral_time(t: f32) -> array<f32, 6> {
  return spiral_embed(3, t * 2.0 * 3.141592);
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = uv;
  return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let spiral_time = get_spiral_time(control.x);  // control.x = t ∈ [0, 1]
    let spiral_pos = get_spiral_pos(uv);

    let x_coord = i32(uv.x * 128.0);
    let y_coord = i32((1 - uv.y) * 240.0);
    let latent = textureLoad(latentTex, vec2<i32>(x_coord, y_coord), 0);
    
    let r = 0.5 + 0.5 * sin(spiral_pos[0] * 2.0 + spiral_time[0] * 4.0 + latent.r * 2.0);
    let g = 0.5 + 0.5 * sin(spiral_pos[6] * 3.0 + spiral_time[3] * 6.0 + latent.g * 2.0);
    let b = 0.5 + 0.5 * sin(spiral_pos[4] * 4.0 + spiral_time[5] * 2.0 + latent.b * 2.0);

    return vec4<f32>(latent.r, latent.g, latent.b, latent.a);
}
