// cpu_encodings.js

export function computeHelixEncoding(coords, numHarmonics) {
    const inArr = Array.isArray(coords) ? coords : [coords];
    const out = [];
    for (let i = 1; i <= numHarmonics; ++i) {
        // 1) all sin(i*coords)
        for (const c of inArr) {
        out.push(Math.sin(i * c));
        }
        // 2) then all cos(i*coords)
        for (const c of inArr) {
        out.push(Math.cos(i * c));
        }
    }
    return out;  // length = 2 * N * numHarmonics
}

export function computeSpiralEncoding(coords, numHarmonics) {
    const inArr = Array.isArray(coords) ? coords : [coords];
    const out = [];
    for (let i = 1; i <= numHarmonics; ++i) {
        // 1) all sin(i*coords)/i
        for (const c of inArr) {
            out.push(Math.sin(i * c) / i);
        }
        // 2) then all cos(i*coords)/i
        for (const c of inArr) {
            out.push(Math.cos(i * c) / i);
        }
    }
    return out;
  }

export function computeSinusoidalEncoding(coords, numHarmonics) {
    // coords: array of length N
    const inArr = Array.isArray(coords) ? coords : [coords];
    const out = [];
    const tau = Math.PI * 2;
    for (let i = 1; i <= numHarmonics; ++i) {
      // first the sine-vector, then the cosine-vector
        for (const c of inArr) {
            out.push(Math.sin(tau * i * c));
        }
        for (const c of inArr) {
            out.push(Math.cos(tau * i * c));
        }
    }
    return out;  // length = N * 2 * numHarmonics
  }
  
export function computeTargetedEncodings(
    xArr,               // [u, v]
    targetDim,          // e.g. 8
    {
        scheme = "sinusoidal",
        norm2Pi = false,
        includeNorm = false,
        includeRaw = false
    } = {}
) {
    const N = xArr.length;
    let out = [];
  
    // 1) raw coords
    if (includeRaw) out.push(...xArr);
  
    // 2) (optional) scale by 2π
    let coords = xArr;
    if (norm2Pi) {
        coords = xArr.map(v => v * 2 * Math.PI);
        if (includeNorm) out.push(...coords);
    }
  
    const rawCount   = includeRaw ? N : 0;
    const baseSlots  = targetDim - rawCount;
    const numHarm    = Math.floor(baseSlots / 2) + 1;

    // 3) encoding
    if (scheme === "sinusoidal") {
        const enc = computeSinusoidalEncoding(coords, numHarm);
        out.push(...enc);  
    } else if (scheme === "helix") {
        const enc = computeHelixEncoding(coords, numHarm);
        out.push(...enc);
    } else if (scheme === "spiral") {
        const enc = computeSpiralEncoding(coords, numHarm);
        out.push(...enc);
    } else if (scheme == null) {
        out.push(...new Array(targetDim - out.length).fill(0));
    } else {
        throw new Error(`Unknown scheme: ${scheme}`);
    }
  
    // 4) trim to targetDim
    return out.slice(0, targetDim);
  }

export function buildLayerMap(manifest /* array from JSON */) {
    const map = new Map();                 // name → {shape, offset,floatOffset}
    manifest.forEach(l => {
        const floatOffset = l.offset >>> 2;  // bytes → float index
        map.set(l.name, {
            shape       : l.shape,             // e.g. [64,12] or [64]
            offsetBytes : l.offset,
            offsetFloats: floatOffset
        });
    });
    return map;
}

export function getWeight(layerMap, floatBuf, layerName, ...indices) {
    const meta = layerMap.get(layerName);
    if (!meta) throw new Error(`layer '${layerName}' not found`);
  
    const {shape, offsetFloats} = meta;
  
    // ----- compute linear index inside the layer -----
    let lin;
    if (shape.length === 1) {
        const i = indices[0] ?? 0;
        lin = i;
    } else if (shape.length === 2) {
        const [rows, cols] = shape;
        const r = indices[0] ?? 0;
        const c = indices[1] ?? 0;
        lin = r * cols + c;                 // row-major layout (PyTorch default)
    } else {
        throw new Error("helper only covers 1-D/2-D tensors");
    }
  
    return floatBuf[offsetFloats + lin];
}

export function computeTrunkAt(x, y, t, ctx) {
    const { floatBuffer, manifest, latentData, width, height } = ctx;
    const modelStart = Math.min(...manifest.filter(l => l.name!=="shared_latent")
                                        .map(l => l.offset / 4));
    // 1) Sample latent at (x,y)
    const idx = y * width + x;
    const base = idx*4;
    const latentPix = latentData.slice(base, base+4);
  
    // 2) Build posEnc
    const u = x/width, v = y/height;
    const pi2 = Math.PI*2;
    const posEnc = [
        u, v,
        Math.sin(pi2*u), Math.sin(pi2*v),
        Math.cos(pi2*u), Math.cos(pi2*v),
        Math.sin(pi2*2*u), Math.sin(pi2*2*v)
    ];
  
    // 3) mainInput = [latentPix, ...posEnc]
    const mainInput = new Float32Array(12);
    mainInput.set(latentPix, 0);
    mainInput.set(posEnc,    4);
  
    // 4) Look up our layer 0 offsets from manifest
    const wMeta = manifest.find(l => l.name.endsWith("layers.0.weight"));
    const bMeta = manifest.find(l => l.name.endsWith("layers.0.bias"));
    if (!wMeta||!bMeta) throw new Error("Can't find layer0 in manifest");
  
    const W0off = wMeta.offset/4 - modelStart; // float index
    const B0off = bMeta.offset/4 - modelStart;
  
    // 5) Run linear 12→64
    const trunk = new Float32Array(64);
    for (let i = 0; i < 64; ++i) {
        let s = 0;
        const rowOff = W0off + i*12;
        for (let j = 0; j < 12; ++j) {
            s += floatBuffer[rowOff + j] * mainInput[j];
        }
        trunk[i] = s + floatBuffer[B0off + i];
    }
  
    // 6) Return the first 8 dims
    return Array.from(trunk.subarray(0,8));
  }
  