from kernels import EnemyKernelDefinition

ENEMY_REGISTRY = {
    "bigChaser": EnemyKernelDefinition(
        targeting="simple_position",
        attacks=["intersect"],
        radius=1.0,
        scale=0.25,
        baseSpeed=0.01,
        maxSpeed=0.01,
        acceleration=0.01,
        rotationSpeed=0.1,
        material="neural_shader",
    ),

}
