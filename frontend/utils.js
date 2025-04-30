export function computePositionOnSphere(radius, theta, phi) {
  return new BABYLON.Vector3(
    radius * Math.sin(phi) * Math.cos(theta), // X
    radius * Math.cos(phi),                  // Y
    radius * Math.sin(phi) * Math.sin(theta) // Z
  );
}
