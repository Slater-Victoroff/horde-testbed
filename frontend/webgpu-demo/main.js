document.addEventListener("DOMContentLoaded", async () => {
   const canvas = document.getElementById("renderCanvas");
   const engine = new BABYLON.WebGPUEngine(canvas, { antialias: true });
   await engine.initAsync();
   console.log("WebGPU engine initialized");
});
