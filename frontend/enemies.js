import { enemyKernels, worldSetupConfig } from './network.js';
import { createNeuralShaderMaterial } from './shaders.js';

export async function spawnEnemy(id, position, kind, scene, enemies, shadowGenerator) {
    const kernel = enemyKernels[kind];
    if (!kernel) {
        console.error(`Kernel not found for enemy kind: ${kind}`);
        return;
    }

    const initialPosition = new BABYLON.Vector3(position[0], kernel.radius, position[2]);

    const enemyMesh = BABYLON.MeshBuilder.CreateIcoSphere(`enemy_${id}`, {
        radius: kernel.radius,
        subdivisions: 4, // Higher subdivisions = smoother sphere
    }, scene);

    if (kernel.material === "neural_shader") {
        // Create the shader material
        const shaderMaterial = await createNeuralShaderMaterial(scene, kernel);
        if (!shaderMaterial) {
            console.error("Failed to create shader material for enemy.");
            return;
        }

        // Assign the shader material to the mesh
        enemyMesh.material = shaderMaterial;
    } else {
        // Default material behavior
        const enemyMaterial = new BABYLON.StandardMaterial(`enemyMaterial_${id}`, scene);

        // Load textures
        const diffuseTexture = new BABYLON.Texture(`http://3.226.12.50:8000/assets/${kernel.material}/albedo.png`, scene);
        const bumpTexture = new BABYLON.Texture(`http://3.226.12.50:8000/assets/${kernel.material}/normal.png`, scene);

        // Adjust texture tiling based on surface area
        const surfaceArea = 4 * Math.PI * Math.pow(kernel.radius, 2); // Surface area of a sphere
        let textureScale = 4 * Math.sqrt(surfaceArea) / (4 * Math.PI); // Proportional scaling
        textureScale = Math.pow(2, Math.round(Math.log2(textureScale))); // Round to nearest power of 2

        diffuseTexture.uScale = textureScale;
        diffuseTexture.vScale = textureScale;
        bumpTexture.uScale = textureScale;
        bumpTexture.vScale = textureScale;
        bumpTexture.level = 5;

        // Assign textures to the material
        enemyMaterial.diffuseTexture = diffuseTexture;
        enemyMaterial.diffuseTexture.coordinatesMode = BABYLON.Texture.TRIPLANAR_MODE;
        enemyMaterial.bumpTexture = bumpTexture;
        enemyMaterial.specularColor = new BABYLON.Color3(0.5, 0.5, 0.5);
        enemyMaterial.ambientColor = new BABYLON.Color3(0.2, 0.2, 0.2);

        enemyMesh.material = enemyMaterial;
    }
    enemyMesh.position = initialPosition;

    shadowGenerator.addShadowCaster(enemyMesh);

    enemies[id] = {
        mesh: enemyMesh,
        position: initialPosition, // Enemy's position
        velocity: new BABYLON.Vector3(0, 0, 0), // Enemy's velocity
        acceleration: new BABYLON.Vector3(0, 0, 0), // Enemy's acceleration
        kind: kind,
        kernel: kernel,
    };
}

function handleTargeting(enemy, player) {
    if (enemy.kernel.targeting === "simple_position") {
        const toPlayer = player.position.subtract(enemy.position);
        toPlayer.y = 0; // Ignore vertical distance

        if (toPlayer.lengthSquared() > 0.01) {
            toPlayer.normalize().scaleInPlace(enemy.kernel.acceleration);
            enemy.acceleration.copyFrom(toPlayer); // Set acceleration toward the player
        } else {
            enemy.acceleration.set(0, 0, 0); // Stop accelerating if very close
        }
    }
}

function updateEnemyPhysics(enemy, player) {
    // Update velocity based on acceleration
    enemy.velocity.addInPlace(enemy.acceleration);

    // Apply friction
    enemy.velocity.scaleInPlace(worldSetupConfig.friction);

    // Cap velocity to max speed
    if (enemy.velocity.length() > enemy.kernel.maxSpeed) {
        enemy.velocity.normalize().scaleInPlace(enemy.kernel.maxSpeed);
    }

    // Update position based on velocity
    enemy.position.addInPlace(enemy.velocity);
    enemy.position.y = enemy.kernel.radius; // Lock Y-axis
    enemy.mesh.position.copyFrom(enemy.position);

    // Smoothly rotate enemy to face the direction of movement
    const toPlayer = player.position.subtract(enemy.position);
    if (toPlayer.lengthSquared() > 0.01) { // Avoid unnecessary calculations for very small velocities
        const targetRotation = Math.atan2(toPlayer.x, toPlayer.z) + Math.PI / 2; // Adjust target angle by 90 degrees
        const currentRotation = enemy.mesh.rotation.y;

        let rotationDiff = targetRotation - currentRotation;
        rotationDiff = ((rotationDiff + Math.PI) % (2 * Math.PI)) - Math.PI;

        // Smoothly rotate toward the target rotation
        enemy.mesh.rotation.y += rotationDiff * enemy.kernel.rotationSpeed;
    }
}


function resolveAllCollisions(enemyList) {
    for (let i = 0; i < enemyList.length; i++) {
        const enemyA = enemyList[i];
        for (let j = i + 1; j < enemyList.length; j++) {
            const enemyB = enemyList[j];

            // Check for collision
            const toOther = enemyB.position.subtract(enemyA.position);
            const minDistance = enemyA.kernel.radius + enemyB.kernel.radius;

            if (toOther.lengthSquared() < minDistance * minDistance) {
                // Resolve collision by pushing enemies apart
                const overlap = minDistance - toOther.length();
                toOther.normalize().scaleInPlace(overlap / 2); // Push them apart equally
                enemyA.position.subtractInPlace(toOther);
                enemyB.position.addInPlace(toOther);

                // Lock Y-axis and update mesh positions
                enemyA.position.y = enemyA.kernel.radius;
                enemyB.position.y = enemyB.kernel.radius;
                enemyA.mesh.position.copyFrom(enemyA.position);
                enemyB.mesh.position.copyFrom(enemyB.position);
            }
        }
    }
}

export function updateEnemies(enemies, player) {
    const enemyList = Object.values(enemies);
    let playerDamaged = false;

    // Update each enemy
    enemyList.forEach((enemy) => {
        handleTargeting(enemy, player);
        updateEnemyPhysics(enemy, player);

        // Check for collisions with the player
        const toPlayer = player.position.subtract(enemy.position);
        toPlayer.y = 0; // Ignore vertical distance
        const collisionThreshold = enemy.kernel.radius + player.radius;

        if (toPlayer.lengthSquared() < collisionThreshold * collisionThreshold) { // Collision threshold
            player.health -= enemy.kernel.scale; // Damage the player
            playerDamaged = true;
        }
        });

        // Resolve collisions between enemies
    resolveAllCollisions(enemyList);

    return playerDamaged;
}

export function exportEnemies(enemies) {
    const enemyData = Object.values(enemies).map(({ mesh, velocity, acceleration }) => [
        ...mesh.position.asArray(),
        ...velocity.asArray(),
        ...acceleration.asArray(),
    ]);

    const kinds = Object.values(enemies).map(({ kind }) => kind);

    return { enemyData, kinds };
}
