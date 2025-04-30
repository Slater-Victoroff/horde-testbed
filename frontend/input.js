export const keys = {};
export const mousePosition = new BABYLON.Vector3(0, 0, 0); // Track mouse position

export function setupInput(canvas, player, onShoot, scene) {
  window.addEventListener("keydown", (e) => keys[e.key.toLowerCase()] = true);
  window.addEventListener("keyup", (e) => keys[e.key.toLowerCase()] = false);

  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    onShoot();
  });

  canvas.addEventListener("mousemove", (event) => {
    const pickResult = scene.pick(event.clientX, event.clientY); // Get 3D position
    if (pickResult.hit) {
      mousePosition.copyFrom(pickResult.pickedPoint);
    }
  });
}
