@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<uniform> control: array<f32, 8>;
@group(0) @binding(2) var latentTex: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let sample = textureSampleLevel(latentTex, sampler(default), uv, 0.0);
  let debug_color = vec4<f32>(
    abs(control[1]), // sin(t)
    abs(control[2]), // cos(t)
    sample.r,        // latent texture red
    1.0
  );
  return debug_color;
}
