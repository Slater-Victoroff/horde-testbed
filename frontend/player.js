import { computePositionOnSphere } from './utils.js';
import { keys, mousePosition } from './input.js';
import { worldSetupConfig } from './network.js';

export function createPlayer(scene) {
  const playerMaterial = new BABYLON.StandardMaterial("playerMaterial", scene);
  playerMaterial.diffuseColor = new BABYLON.Color3(0.4, 0.8, 1);

  const playerRadius = 0.5;
  const playerMesh = BABYLON.MeshBuilder.CreateSphere("player", { diameter: 2 * playerRadius }, scene);
  playerMesh.material = playerMaterial;
  playerMesh.position.y = 1.1 * playerRadius;

  // Add "eyes" and "mouth"
  const eyeMaterial = new BABYLON.StandardMaterial("eyeMaterial", scene);
  eyeMaterial.diffuseColor = new BABYLON.Color3(0, 0, 0);

  const leftEye = BABYLON.MeshBuilder.CreateSphere("leftEye", { diameter: 0.2 }, scene);
  leftEye.material = eyeMaterial;
  leftEye.parent = playerMesh;
  leftEye.position = computePositionOnSphere(playerRadius, Math.PI / 4, Math.PI / 3);

  const rightEye = BABYLON.MeshBuilder.CreateSphere("rightEye", { diameter: 0.2 }, scene);
  rightEye.material = eyeMaterial;
  rightEye.parent = playerMesh;
  rightEye.position = computePositionOnSphere(playerRadius, 3 * Math.PI / 4, Math.PI / 3);

  const mouthMaterial = new BABYLON.StandardMaterial("mouthMaterial", scene);
  mouthMaterial.diffuseColor = new BABYLON.Color3(1, 0, 0);

  const mouth = BABYLON.MeshBuilder.CreateDisc("mouth", { radius: 0.2, tessellation: 32 }, scene);
  mouth.material = mouthMaterial;
  mouth.parent = playerMesh;
  mouth.position = computePositionOnSphere(playerRadius, Math.PI / 2, Math.PI / 2);
  mouth.rotation.x = BABYLON.Tools.ToRadians(90);

  return {
    mesh: playerMesh,
    position: new BABYLON.Vector3(0, playerRadius, 0), // Player's position
    velocity: new BABYLON.Vector3(0, 0, 0), // Player's velocity
    acceleration: new BABYLON.Vector3(0, 0, 0), // Player's acceleration
    radius: playerRadius,
    speed: 0.1,
    accelerationRate: 0.15,
    maxSpeed: 0.2,
    rotationSpeed: 0.1,
    health: 100,
    attack: 10,
    defense: 5,
    level: 1,
    experience: 0,
    inventory: [],
  };
}

function handleInput(player) {
  // Update acceleration based on keyboard input
  const inputDir = new BABYLON.Vector3(
    (keys["d"] ? 1 : 0) - (keys["a"] ? 1 : 0),
    0,
    (keys["w"] ? 1 : 0) - (keys["s"] ? 1 : 0)
  );

  if (inputDir.lengthSquared() > 0) {
    inputDir.normalize().scaleInPlace(player.speed);
    player.acceleration.copyFrom(inputDir).scaleInPlace(player.accelerationRate);
  } else {
    player.acceleration.set(0, 0, 0); // No input, no acceleration
  }
}

function updateRotation(player) {
  // Rotate the player to face the mouse position
  const toMouse = mousePosition.subtract(player.mesh.position);
  if (toMouse.lengthSquared() > 0.01) { // Avoid jitter when mouse is too close
    const targetRotation = Math.atan2(toMouse.x, toMouse.z);
    const currentRotation = player.mesh.rotation.y;

    let rotationDiff = targetRotation - currentRotation;
    rotationDiff = ((rotationDiff + Math.PI) % (2 * Math.PI)) - Math.PI;

    // Smoothly rotate toward the target rotation
    player.mesh.rotation.y += rotationDiff * player.rotationSpeed;
  }
}

export function updatePlayer(player) {
  // Handle input to update acceleration
  handleInput(player);
  // Update velocity based on acceleration
  player.velocity.addInPlace(player.acceleration);

  // Apply friction
  player.velocity.scaleInPlace(worldSetupConfig.friction);

  // Cap velocity to max speed
  if (player.velocity.length() > player.maxSpeed) {
    player.velocity.normalize().scaleInPlace(player.maxSpeed);
  }

  // Update position based on velocity
  player.position.addInPlace(player.velocity);
  player.mesh.position.copyFrom(player.position);

  // Update rotation to face the mouse
  updateRotation(player);
}
