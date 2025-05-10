@group(0) @binding(0) var<uniform> control: vec4<f32>;
@group(0) @binding(1) var latentTex: texture_2d<f32>;
@group(0) @binding(2) var latentSampler: sampler;
@group(0) @binding(3) var<storage, read> modelBuf: array<f32>;
@group(0) @binding(4) var<storage, read_write> debugOut: array<f32>;

struct ModelMeta {
    start: u32,
};

@group(0) @binding(5) var<uniform> modelMeta: ModelMeta;
@group(0) @binding(6) var<storage, read> layerOffsets: array<u32>;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
};


// ── 1) a simple 2D hash → [0,1) based on uv + time ────────────────────
fn rand(p: vec2<f32>) -> f32 {
    // these magic constants give a decent pseudo-random spread
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}


fn uv_to_texCoord(uv: vec2<f32>) -> vec2<i32> {
    // textureDimensions on a sampled texture_2d returns vec2<u32>
    let dims: vec2<u32> = textureDimensions(latentTex);
    return vec2<i32>(
        i32(uv.x * f32(dims.x)),
        i32(uv.y * f32(dims.y))
    );
}


fn getWeightBase(layerIndex: u32) -> u32 {
  return layerOffsets[layerIndex * 2u + 0u];
}


fn getBiasBase(layerIndex: u32) -> u32 {
  return layerOffsets[layerIndex * 2u + 1u];
}

fn relu(x: f32) -> f32 { return max(x, 0.0); }


fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159265359) * (x + 0.044715 * x * x * x)));
}


fn relu_vec(input: array<f32, 32>) -> array<f32, 32> {
    var out: array<f32, 32>;
    for (var i = 0; i < 32; i = i + 1) {
        out[i] = relu(input[i]);
    }
    return out;
}


fn gelu_vec(input: array<f32, 64>) -> array<f32, 64> {
    var out: array<f32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        out[i] = gelu(input[i]);
    }
    return out;
}


fn sigmoid_vec4(v: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-v));
}


fn get_pos_enc(uv: vec2<f32>) -> array<f32, 8> {
    let x = uv.x;
    let y = uv.y;
    let tau = 6.28318530718;

    return array<f32,8>(
        x, y,
        sin(tau * x), sin(tau * y),
        cos(tau * x), cos(tau * y),
        sin(tau * 2.0 * x), sin(tau * 2.0 * y)
    );
}

fn get_spiral_pos(coords: vec2<f32>) -> array<f32,16> {
    let tau = 6.28318530718;
    let scaled_x = coords.x * tau;
    let scaled_y = coords.y * tau;
    var out: array<f32,16>;
    // 0–1: include_norm
    out[0] = scaled_x;
    out[1] = scaled_y;

    // harmonics 1..3 (4 values each)
    let inv1 = 1.0;
    out[2] = sin(1.0 * scaled_x) * inv1;
    out[3] = sin(1.0 * scaled_y) * inv1;
    out[4] = cos(1.0 * scaled_x) * inv1;
    out[5] = cos(1.0 * scaled_y) * inv1;

    let inv2 = 0.5;
    out[6]  = sin(2.0 * scaled_x) * inv2;
    out[7]  = sin(2.0 * scaled_y) * inv2;
    out[8]  = cos(2.0 * scaled_x) * inv2;
    out[9]  = cos(2.0 * scaled_y) * inv2;

    let inv3 = 1.0 / 3.0;
    out[10] = sin(3.0 * scaled_x) * inv3;
    out[11] = sin(3.0 * scaled_y) * inv3;
    out[12] = cos(3.0 * scaled_x) * inv3;
    out[13] = cos(3.0 * scaled_y) * inv3;

    // 4th harmonic: only take the two sine terms
    let inv4 = 0.25;
    out[14] = sin(4.0 * scaled_x) * inv4;
    out[15] = sin(4.0 * scaled_y) * inv4;

    return out;
}

// WGSL version of compute_targeted_encodings(
//    x=[[t]], target_dim=8,
//    scheme="spiral", norm2pi=true, include_norm=true, include_raw=true
// )
fn get_spiral_time(t: f32) -> array<f32, 8> {
    let tau = 6.28318530718;
    let x   = t * tau;          // scaled time

    var out: array<f32, 8>;
    // 0) raw t
    out[0] = t;
    // 1) normed = t * 2π
    out[1] = x;

    // Now spiral harmonics i=1..3 (we only need 3 to fill up to 8 elements)
    // 2) i=1
    out[2] = sin(1.0 * x) / 1.0;
    out[3] = cos(1.0 * x) / 1.0;
    // 3) i=2
    out[4] = sin(2.0 * x) / 2.0;
    out[5] = cos(2.0 * x) / 2.0;
    // 4) i=3
    out[6] = sin(3.0 * x) / 3.0;
    out[7] = cos(3.0 * x) / 3.0;

    return out;
}

fn get_main_input(latent: vec4<f32>, pos_enc: array<f32, 8>) -> array<f32, 12> {
    var out: array<f32, 12>;
    out[0] = latent.r;
    out[1] = latent.g;
    out[2] = latent.b;
    out[3] = latent.a;
    for (var i = 0; i < 8; i = i + 1) {
        out[4 + i] = pos_enc[i];
    }
    return out;
}

fn load_scalar(index : u32) -> f32 {
    // index is *global* float index from the manifest
    let rel : i32 = i32(index) - i32(modelMeta.start);   // signed
    return modelBuf[rel];
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = uv;
  return out;
}


fn linear_12x64(input: array<f32, 12>, weight_offset: u32, bias_offset: u32) -> array<f32, 64> {
    var out: array<f32, 64>;

    for (var i : u32 = 0u; i < 64u; i = i + 1u) {
        var sum = 0.0;
        for (var j : u32 = 0u; j < 12u; j = j + 1u) {
            let w = load_scalar(weight_offset + i * 12u + j);
            sum = sum + (w * input[j]);
        }
        let b = load_scalar(bias_offset + i);
        out[i] = sum + b;
    }

    return out;
}

fn get_film_input(spiral_time: array<f32, 8>, spiral_pos: array<f32, 12>) -> array<f32, 20> {
    var out: array<f32, 20>;
    for (var i = 0; i < 8; i = i + 1) {
        out[i] = spiral_time[i];
    }
    for (var i = 0; i < 12; i = i + 1) {
        out[8 + i] = spiral_pos[i];
    }
    return out;
}

fn linear_8x32(inp: array<f32, 8>, weight_offset: u32, bias_offset: u32) -> array<f32, 32> {
    var o: array<f32, 32>;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        var s = 0.0;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            s += load_scalar(weight_offset + i * 8u + j) * inp[j];
        }
        o[i] = s + load_scalar(bias_offset + i);
    }
    return o;
}

fn linear_16x32(inp: array<f32, 16>, weight_offset: u32, bias_offset: u32) -> array<f32, 32> {
    var o: array<f32, 32>;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        var s = 0.0;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            s += load_scalar(weight_offset + i * 16u + j) * inp[j];
        }
        o[i] = s + load_scalar(bias_offset + i);
    }
    return o;
}

fn linear_64x128(input: array<f32, 64>, weight_offset: u32, bias_offset: u32) -> array<f32, 128> {
    var out: array<f32, 128>;
    for (var i: u32 = 0u; i < 128u; i = i + 1u) {
        var sum = 0.0;
        for (var j: u32 = 0u; j < 64u; j = j + 1u) {
            let w = load_scalar(weight_offset + i * 64u + j);
            sum = sum + w * input[j];
        }
        let b = load_scalar(bias_offset + i);
        out[i] = sum + b;
    }
    return out;
}

fn apply_film(x: array<f32, 64>, gamma: array<f32, 64>, beta: array<f32, 64>) -> array<f32, 64> {
    var out: array<f32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        out[i] = (gamma[i] * x[i]) + beta[i];
    }
    return out;
}

fn linear_64x64(input: array<f32, 64>, weight_offset: u32, bias_offset: u32) -> array<f32, 64> {
    var out: array<f32, 64>;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        var sum = 0.0;
        for (var j: u32 = 0u; j < 64u; j = j + 1u) {
            let w = load_scalar(weight_offset + i * 64u + j);
            sum = sum + (w * input[j]);
        }
        let b = load_scalar(bias_offset + i);
        out[i] = sum + b;
    }
    return out;
}

fn linear_64x4(input: array<f32, 64>, weight_offset: u32, bias_offset: u32) -> vec4<f32> {
    var out: vec4<f32> = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        var sum = 0.0;
        for (var j: u32 = 0u; j < 64u; j = j + 1u) {
            let w = load_scalar(weight_offset + i * 64u + j);
            sum = sum + (w * input[j]);
        }
        let b = load_scalar(bias_offset + i);
        out[i] = sum + b;
    }
    return out;
}

const TIME_LAYER : u32 = 0u;
const POS_LAYER  : u32 = 2u;
const FILM_LAYER : u32 = 4u;
const L0_LAYER   : u32 = 6u;
const L2_LAYER   : u32 = 8u;
const L3_LAYER   : u32 = 10u;
const L4_LAYER   : u32 = 12u;
const roamAmp    : f32 = 0.4;
const roamRange  : f32 = 0.2;
const tau = 6.28318530718;
const harmFreqs  : array<f32,3> = array<f32,3>( 1.0, 2.0, 3.0 );
const harmOffs   : array<f32,3> = array<f32,3>( 0.50, 0.25, 0.50 );
const harmSpans  : array<f32,3> = array<f32,3>( 0.30, 0.50, 0.40 );

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let timeNode = 0.27;
    let pos_enc: array<f32, 8> = get_pos_enc(uv);
    let spiral_pos = get_spiral_pos(uv);

    let tCont : f32 = control.x;   // grows linearly, e.g. elapsed*roamSpeed

    let oscillation : f32 = sin(tCont * tau) * (roamRange * 0.5);

    let newTime : f32 = timeNode + oscillation;

    let wrappedT : f32 = fract(newTime);

    var dynEnc : array<f32,8> = get_spiral_time(wrappedT);
    var staticEnc : array<f32, 8> = get_spiral_time(timeNode);

    var enc : array<f32, 8>;
    enc[0] = staticEnc[0];
    enc[1] = staticEnc[1];
    enc[2] = staticEnc[2];
    enc[3] = staticEnc[3];
    // let mid2 = fract( (control.x * 2.0) + 0.25 ); // map 0→½ into [¼,¾]
    // then overwrite only slot 4/5:
    // enc[4] = sin(2.0 * tau * mid2) / 2.0;
    // enc[5] = cos(2.0 * tau * mid2) / 2.0;
    // enc[6] = staticEnc[6];
    // enc[7] = staticEnc[7];

    for (var h = 1u; h < 3u; h = h + 1u) {
        let freq    = harmFreqs[h];    // e.g. 2.0, 3.0
        let center  = harmOffs[h];     // center of your mini‐loop
        let width   = harmSpans[h];    // total span of mini‐loop

        // 1) continuous phase for this harmonic
        let phase   = tCont * freq;                // tCont from JS: elapsed*roamSpeed
        // 2) sine‐wave in [-1,+1], then scale to [–width/2, +width/2]
        let wobble  = sin(phase * tau) * (width * 0.5);
        // 3) offset around center
        let midT    = center + wobble;             // now lives in [center−w/2, center+w/2]
        // 4) compute that harmonic’s sin/cos
        let phi     = midT * tau * freq;
        let idx     = 2u*h + 2u;                   // 4 for h=1, 6 for h=2

        let baseSin = sin(phi) / freq;
        let baseCos = cos(phi) / freq;
        // small amplitude jitter in [0.9,1.1]
        let aJitter = 1.0 + sin(tCont * tau * freq)*0.05;
        enc[idx]   = baseSin * aJitter;
        enc[idx+1] = baseCos * aJitter;
        // enc[idx  ] = sin(phi) / freq;
        // enc[idx+1] = cos(phi) / freq;
    }

    let lat = textureLoad(latentTex, uv_to_texCoord(uv), 0);  // no 1.0-y flip!

    /* 3b. assemble 12-vector & run first linear */
    let main_in  = get_main_input(lat, pos_enc);
    let trunk  = linear_12x64(main_in, getWeightBase(L0_LAYER), getBiasBase(L0_LAYER));

    var time_embed  = linear_8x32(enc, getWeightBase(TIME_LAYER), getBiasBase(TIME_LAYER));
    time_embed = relu_vec(time_embed);

    var pos_embed  = linear_16x32(spiral_pos, getWeightBase(POS_LAYER), getBiasBase(POS_LAYER));
    pos_embed = relu_vec(pos_embed);

    var film_in: array<f32, 64>;
    for (var i=0;i<32;i++){ film_in[i]     = pos_embed[i]; }
    for (var i=0;i<32;i++){ film_in[32+i]  = time_embed[i]; }

    let film_out = linear_64x128(film_in, getWeightBase(FILM_LAYER), getBiasBase(FILM_LAYER));

    var gamma: array<f32, 64>;
    var beta: array<f32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        gamma[i] = film_out[i];
        beta[i] = film_out[i + 64];
    }

    let modulated = apply_film(trunk, gamma, beta);
    let activated = gelu_vec(modulated);

    let after2 = linear_64x64(activated, getWeightBase(L2_LAYER), getBiasBase(L2_LAYER));
    let activated2 = gelu_vec(after2);

    let after3 = linear_64x64(activated2, getWeightBase(L3_LAYER), getBiasBase(L3_LAYER));
    let activated3 = gelu_vec(after3);

    let raw_output = linear_64x4(activated3, getWeightBase(L4_LAYER), getBiasBase(L4_LAYER));

    let final_color = sigmoid_vec4(raw_output);

    return final_color;
}


