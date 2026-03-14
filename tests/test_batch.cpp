// Copyright (c) Amer Koleci and Contributors.
// Licensed under the MIT License (MIT). See LICENSE in the repository root for more information.
// Batch API tests for joltc

#include <gtest/gtest.h>
#include "joltc.h"

namespace BroadPhaseLayers {
	static constexpr JPH_BroadPhaseLayer NON_MOVING = 0;
	static constexpr JPH_BroadPhaseLayer MOVING = 1;
	static constexpr uint32_t NUM_LAYERS = 2;
}

namespace ObjectLayers {
	static constexpr JPH_ObjectLayer NON_MOVING = 0;
	static constexpr JPH_ObjectLayer MOVING = 1;
	static constexpr uint32_t NUM_LAYERS = 2;
}

class BatchTest : public ::testing::Test {
protected:
	JPH_PhysicsSystem* physicsSystem = nullptr;
	JPH_BodyInterface* bodyInterface = nullptr;
	JPH_JobSystem* jobSystem = nullptr;

	void SetUp() override {
		ASSERT_TRUE(JPH_Init());
		jobSystem = JPH_JobSystemThreadPool_Create(nullptr);
		ASSERT_NE(jobSystem, nullptr);

		auto bpLayerInterface = JPH_BroadPhaseLayerInterfaceTable_Create(
			ObjectLayers::NUM_LAYERS, BroadPhaseLayers::NUM_LAYERS);
		JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
			bpLayerInterface, ObjectLayers::NON_MOVING, BroadPhaseLayers::NON_MOVING);
		JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
			bpLayerInterface, ObjectLayers::MOVING, BroadPhaseLayers::MOVING);

		auto objectLayerPairFilter = JPH_ObjectLayerPairFilterTable_Create(ObjectLayers::NUM_LAYERS);
		JPH_ObjectLayerPairFilterTable_EnableCollision(objectLayerPairFilter, ObjectLayers::NON_MOVING, ObjectLayers::MOVING);
		JPH_ObjectLayerPairFilterTable_EnableCollision(objectLayerPairFilter, ObjectLayers::MOVING, ObjectLayers::MOVING);

		auto objectVsBroadPhaseLayerFilter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
			bpLayerInterface, BroadPhaseLayers::NUM_LAYERS,
			objectLayerPairFilter, ObjectLayers::NUM_LAYERS);

		JPH_PhysicsSystemSettings settings = {};
		settings.maxBodies = 1024;
		settings.numBodyMutexes = 0;
		settings.maxBodyPairs = 1024;
		settings.maxContactConstraints = 1024;
		settings.broadPhaseLayerInterface = (JPH_BroadPhaseLayerInterface*)bpLayerInterface;
		settings.objectLayerPairFilter = (JPH_ObjectLayerPairFilter*)objectLayerPairFilter;
		settings.objectVsBroadPhaseLayerFilter = (JPH_ObjectVsBroadPhaseLayerFilter*)objectVsBroadPhaseLayerFilter;

		physicsSystem = JPH_PhysicsSystem_Create(&settings);
		ASSERT_NE(physicsSystem, nullptr);

		bodyInterface = JPH_PhysicsSystem_GetBodyInterface(physicsSystem);
		ASSERT_NE(bodyInterface, nullptr);
	}

	void TearDown() override {
		if (physicsSystem) JPH_PhysicsSystem_Destroy(physicsSystem);
		if (jobSystem) JPH_JobSystem_Destroy(jobSystem);
		JPH_Shutdown();
	}

	JPH_BodyID CreateDynamicSphere(float x, float y, float z, float radius = 0.5f) {
		JPH_SphereShape* shape = JPH_SphereShape_Create(radius);
		JPH_RVec3 position = {x, y, z};
		JPH_Quat rotation = {0.0f, 0.0f, 0.0f, 1.0f};
		JPH_BodyCreationSettings* settings = JPH_BodyCreationSettings_Create3(
			(JPH_Shape*)shape, &position, &rotation,
			JPH_MotionType_Dynamic, ObjectLayers::MOVING);
		JPH_Body* body = JPH_BodyInterface_CreateBody(bodyInterface, settings);
		JPH_BodyID bodyId = JPH_Body_GetID(body);
		JPH_BodyCreationSettings_Destroy(settings);
		JPH_Shape_Destroy((JPH_Shape*)shape);
		return bodyId;
	}

	JPH_BodyID CreateKinematicSphere(float x, float y, float z, float radius = 0.5f) {
		JPH_SphereShape* shape = JPH_SphereShape_Create(radius);
		JPH_RVec3 position = {x, y, z};
		JPH_Quat rotation = {0.0f, 0.0f, 0.0f, 1.0f};
		JPH_BodyCreationSettings* settings = JPH_BodyCreationSettings_Create3(
			(JPH_Shape*)shape, &position, &rotation,
			JPH_MotionType_Kinematic, ObjectLayers::MOVING);
		JPH_Body* body = JPH_BodyInterface_CreateBody(bodyInterface, settings);
		JPH_BodyID bodyId = JPH_Body_GetID(body);
		JPH_BodyCreationSettings_Destroy(settings);
		JPH_Shape_Destroy((JPH_Shape*)shape);
		return bodyId;
	}

	JPH_BodyID CreateStaticFloor() {
		JPH_Vec3 halfExtent = {100.0f, 0.5f, 100.0f};
		JPH_BoxShape* shape = JPH_BoxShape_Create(&halfExtent, 0.0f);
		JPH_RVec3 position = {0.0f, -0.5f, 0.0f};
		JPH_Quat rotation = {0.0f, 0.0f, 0.0f, 1.0f};
		JPH_BodyCreationSettings* settings = JPH_BodyCreationSettings_Create3(
			(JPH_Shape*)shape, &position, &rotation,
			JPH_MotionType_Static, ObjectLayers::NON_MOVING);
		JPH_Body* body = JPH_BodyInterface_CreateBody(bodyInterface, settings);
		JPH_BodyID bodyId = JPH_Body_GetID(body);
		JPH_BodyInterface_AddBody(bodyInterface, bodyId, JPH_Activation_DontActivate);
		JPH_BodyCreationSettings_Destroy(settings);
		JPH_Shape_Destroy((JPH_Shape*)shape);
		return bodyId;
	}
};

// =============================================================================
// Batch Body Lifecycle Tests
// =============================================================================

TEST_F(BatchTest, AddBodies_BatchAdd) {
	const uint32_t count = 8;
	JPH_BodyID bodyIDs[8];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 10.0f, 0.0f);
	}

	// Batch add
	JPH_BodyInterface_AddBodies(bodyInterface, bodyIDs, count, JPH_Activation_Activate);

	// Verify all bodies are added and active
	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_TRUE(JPH_BodyInterface_IsAdded(bodyInterface, bodyIDs[i]));
		EXPECT_TRUE(JPH_BodyInterface_IsActive(bodyInterface, bodyIDs[i]));
	}

	// Cleanup
	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, RemoveBodies_BatchRemove) {
	const uint32_t count = 5;
	JPH_BodyID bodyIDs[5];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 10.0f, 0.0f);
	}
	JPH_BodyInterface_AddBodies(bodyInterface, bodyIDs, count, JPH_Activation_Activate);

	// Batch remove
	JPH_BodyInterface_RemoveBodies(bodyInterface, bodyIDs, count);

	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FALSE(JPH_BodyInterface_IsAdded(bodyInterface, bodyIDs[i]));
	}

	// Cleanup
	JPH_BodyInterface_DestroyBodies(bodyInterface, bodyIDs, count);
}

TEST_F(BatchTest, DestroyBodies_BatchDestroy) {
	const uint32_t count = 4;
	JPH_BodyID bodyIDs[4];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 10.0f, 0.0f);
	}
	JPH_BodyInterface_AddBodies(bodyInterface, bodyIDs, count, JPH_Activation_Activate);
	JPH_BodyInterface_RemoveBodies(bodyInterface, bodyIDs, count);

	// Batch destroy should not crash
	JPH_BodyInterface_DestroyBodies(bodyInterface, bodyIDs, count);
}

TEST_F(BatchTest, AddBodies_ZeroCount) {
	// Zero count should not crash
	JPH_BodyInterface_AddBodies(bodyInterface, nullptr, 0, JPH_Activation_Activate);
}

TEST_F(BatchTest, AddBodies_SingleBody) {
	JPH_BodyID bodyId = CreateDynamicSphere(0.0f, 10.0f, 0.0f);

	JPH_BodyInterface_AddBodies(bodyInterface, &bodyId, 1, JPH_Activation_Activate);
	EXPECT_TRUE(JPH_BodyInterface_IsAdded(bodyInterface, bodyId));
	EXPECT_TRUE(JPH_BodyInterface_IsActive(bodyInterface, bodyId));

	JPH_BodyInterface_RemoveBody(bodyInterface, bodyId);
	JPH_BodyInterface_DestroyBody(bodyInterface, bodyId);
}

// =============================================================================
// Batch Transform Read/Write Tests
// =============================================================================

TEST_F(BatchTest, GetPositionsAndRotations_BatchRead) {
	const uint32_t count = 4;
	JPH_BodyID bodyIDs[4];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 3.0f, (float)i * 5.0f + 1.0f, (float)i * 7.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_RVec3 positions[4];
	JPH_Quat rotations[4];
	JPH_BodyInterface_GetPositionsAndRotations(bodyInterface, bodyIDs, count, positions, rotations);

	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(positions[i].x, (float)i * 3.0f);
		EXPECT_FLOAT_EQ(positions[i].y, (float)i * 5.0f + 1.0f);
		EXPECT_FLOAT_EQ(positions[i].z, (float)i * 7.0f);

		// Identity rotation
		EXPECT_NEAR(rotations[i].x, 0.0f, 1e-6f);
		EXPECT_NEAR(rotations[i].y, 0.0f, 1e-6f);
		EXPECT_NEAR(rotations[i].z, 0.0f, 1e-6f);
		EXPECT_NEAR(rotations[i].w, 1.0f, 1e-6f);
	}

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, SetPositionsAndRotations_BatchWrite) {
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere(0.0f, 10.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_RVec3 newPositions[3] = {{10.0f, 20.0f, 30.0f}, {40.0f, 50.0f, 60.0f}, {70.0f, 80.0f, 90.0f}};
	JPH_Quat newRotations[3] = {{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}};
	JPH_BodyInterface_SetPositionsAndRotations(bodyInterface, bodyIDs, count, newPositions, newRotations, JPH_Activation_Activate);

	// Verify by reading back
	JPH_RVec3 readPositions[3];
	JPH_Quat readRotations[3];
	JPH_BodyInterface_GetPositionsAndRotations(bodyInterface, bodyIDs, count, readPositions, readRotations);

	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readPositions[i].x, newPositions[i].x);
		EXPECT_FLOAT_EQ(readPositions[i].y, newPositions[i].y);
		EXPECT_FLOAT_EQ(readPositions[i].z, newPositions[i].z);
	}

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

// =============================================================================
// Batch Velocity Read/Write Tests
// =============================================================================

TEST_F(BatchTest, GetLinearAndAngularVelocities_BatchRead) {
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	// Set individual velocities
	JPH_Vec3 vel1 = {1.0f, 0.0f, 0.0f};
	JPH_Vec3 vel2 = {0.0f, 2.0f, 0.0f};
	JPH_Vec3 vel3 = {0.0f, 0.0f, 3.0f};
	JPH_BodyInterface_SetLinearVelocity(bodyInterface, bodyIDs[0], &vel1);
	JPH_BodyInterface_SetLinearVelocity(bodyInterface, bodyIDs[1], &vel2);
	JPH_BodyInterface_SetLinearVelocity(bodyInterface, bodyIDs[2], &vel3);

	JPH_Vec3 linVels[3], angVels[3];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_FLOAT_EQ(linVels[0].x, 1.0f);
	EXPECT_FLOAT_EQ(linVels[1].y, 2.0f);
	EXPECT_FLOAT_EQ(linVels[2].z, 3.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, SetLinearAndAngularVelocities_BatchWrite) {
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 linVels[3] = {{5.0f, 0.0f, 0.0f}, {0.0f, 10.0f, 0.0f}, {0.0f, 0.0f, 15.0f}};
	JPH_Vec3 angVels[3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
	JPH_BodyInterface_SetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	JPH_Vec3 readLinVels[3], readAngVels[3];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, readLinVels, readAngVels);

	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readLinVels[i].x, linVels[i].x);
		EXPECT_FLOAT_EQ(readLinVels[i].y, linVels[i].y);
		EXPECT_FLOAT_EQ(readLinVels[i].z, linVels[i].z);
		EXPECT_FLOAT_EQ(readAngVels[i].x, angVels[i].x);
		EXPECT_FLOAT_EQ(readAngVels[i].y, angVels[i].y);
		EXPECT_FLOAT_EQ(readAngVels[i].z, angVels[i].z);
	}

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

// =============================================================================
// Batch Full State Read/Write Tests
// =============================================================================

TEST_F(BatchTest, GetTransformsAndVelocities_FullState) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 10.0f, 20.0f, 30.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 vel = {1.0f, 2.0f, 3.0f};
	JPH_BodyInterface_SetLinearVelocity(bodyInterface, bodyIDs[0], &vel);

	JPH_RVec3 positions[2];
	JPH_Quat rotations[2];
	JPH_Vec3 linVels[2], angVels[2];
	JPH_BodyInterface_GetTransformsAndVelocities(bodyInterface, bodyIDs, count, positions, rotations, linVels, angVels);

	EXPECT_FLOAT_EQ(positions[0].x, 0.0f);
	EXPECT_FLOAT_EQ(positions[1].x, 10.0f);
	EXPECT_FLOAT_EQ(linVels[0].x, 1.0f);
	EXPECT_FLOAT_EQ(linVels[0].y, 2.0f);
	EXPECT_FLOAT_EQ(linVels[0].z, 3.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, SetPositionRotationAndVelocities_FullState) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere(0.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_RVec3 newPositions[2] = {{100.0f, 200.0f, 300.0f}, {400.0f, 500.0f, 600.0f}};
	JPH_Quat newRotations[2] = {{0, 0, 0, 1}, {0, 0, 0, 1}};
	JPH_Vec3 newLinVels[2] = {{11.0f, 12.0f, 13.0f}, {21.0f, 22.0f, 23.0f}};
	JPH_Vec3 newAngVels[2] = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}};

	JPH_BodyInterface_SetPositionRotationAndVelocities(bodyInterface, bodyIDs, count,
		newPositions, newRotations, newLinVels, newAngVels);

	// Read back and verify
	JPH_RVec3 readPositions[2];
	JPH_Quat readRotations[2];
	JPH_Vec3 readLinVels[2], readAngVels[2];
	JPH_BodyInterface_GetTransformsAndVelocities(bodyInterface, bodyIDs, count,
		readPositions, readRotations, readLinVels, readAngVels);

	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(readPositions[i].x, newPositions[i].x);
		EXPECT_FLOAT_EQ(readPositions[i].y, newPositions[i].y);
		EXPECT_FLOAT_EQ(readPositions[i].z, newPositions[i].z);
		EXPECT_FLOAT_EQ(readLinVels[i].x, newLinVels[i].x);
		EXPECT_FLOAT_EQ(readLinVels[i].y, newLinVels[i].y);
		EXPECT_FLOAT_EQ(readLinVels[i].z, newLinVels[i].z);
		EXPECT_NEAR(readAngVels[i].x, newAngVels[i].x, 1e-5f);
		EXPECT_NEAR(readAngVels[i].y, newAngVels[i].y, 1e-5f);
		EXPECT_NEAR(readAngVels[i].z, newAngVels[i].z, 1e-5f);
	}

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

// =============================================================================
// Batch Force/Impulse Tests
// =============================================================================

TEST_F(BatchTest, AddForces_BatchApply) {
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 forces[3] = {{100.0f, 0.0f, 0.0f}, {0.0f, 200.0f, 0.0f}, {0.0f, 0.0f, 300.0f}};
	JPH_BodyInterface_AddForces(bodyInterface, bodyIDs, count, forces);

	// Step physics to let forces take effect
	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);

	// Bodies should have non-zero velocity after force application
	JPH_Vec3 linVels[3], angVels[3];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_GT(linVels[0].x, 0.0f);
	// Y force may not overcome gravity, but should be greater than a body with no force applied
	// Just verify the X and Z axes where there's no gravity interference
	EXPECT_GT(linVels[2].z, 0.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, AddTorques_BatchApply) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 torques[2] = {{10.0f, 0.0f, 0.0f}, {0.0f, 10.0f, 0.0f}};
	JPH_BodyInterface_AddTorques(bodyInterface, bodyIDs, count, torques);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);

	JPH_Vec3 linVels[2], angVels[2];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_GT(fabsf(angVels[0].x), 0.0f);
	EXPECT_GT(fabsf(angVels[1].y), 0.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, AddForcesAndTorques_BatchApply) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 forces[2] = {{100.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 100.0f}};
	JPH_Vec3 torques[2] = {{0.0f, 0.0f, 5.0f}, {5.0f, 0.0f, 0.0f}};
	JPH_BodyInterface_AddForcesAndTorques(bodyInterface, bodyIDs, count, forces, torques);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);

	JPH_Vec3 linVels[2], angVels[2];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_GT(linVels[0].x, 0.0f);
	EXPECT_GT(linVels[1].z, 0.0f);
	EXPECT_GT(fabsf(angVels[0].z), 0.0f);
	EXPECT_GT(fabsf(angVels[1].x), 0.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, AddImpulses_BatchApply) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 impulses[2] = {{10.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 10.0f}};
	JPH_BodyInterface_AddImpulses(bodyInterface, bodyIDs, count, impulses);

	// Impulses take effect immediately (no step needed for velocity check)
	JPH_Vec3 linVels[2], angVels[2];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_GT(linVels[0].x, 0.0f);
	EXPECT_GT(linVels[1].z, 0.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

TEST_F(BatchTest, AddAngularImpulses_BatchApply) {
	const uint32_t count = 2;
	JPH_BodyID bodyIDs[2];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_Vec3 angImpulses[2] = {{5.0f, 0.0f, 0.0f}, {0.0f, 5.0f, 0.0f}};
	JPH_BodyInterface_AddAngularImpulses(bodyInterface, bodyIDs, count, angImpulses);

	JPH_Vec3 linVels[2], angVels[2];
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, bodyIDs, count, linVels, angVels);

	EXPECT_GT(fabsf(angVels[0].x), 0.0f);
	EXPECT_GT(fabsf(angVels[1].y), 0.0f);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

// =============================================================================
// Batch Kinematic Move Tests
// =============================================================================

TEST_F(BatchTest, MoveKinematics_BatchMove) {
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];

	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateKinematicSphere((float)i * 2.0f, 5.0f, 0.0f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_RVec3 targets[3] = {{10.0f, 5.0f, 0.0f}, {20.0f, 5.0f, 0.0f}, {30.0f, 5.0f, 0.0f}};
	JPH_Quat rotations[3] = {{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}};
	JPH_BodyInterface_MoveKinematics(bodyInterface, bodyIDs, count, targets, rotations, 1.0f / 60.0f);

	// After stepping, kinematic bodies should have moved towards targets
	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);

	JPH_RVec3 positions[3];
	JPH_Quat readRotations[3];
	JPH_BodyInterface_GetPositionsAndRotations(bodyInterface, bodyIDs, count, positions, readRotations);

	// Kinematic bodies should now be at target positions
	for (uint32_t i = 0; i < count; ++i) {
		EXPECT_FLOAT_EQ(positions[i].x, targets[i].x);
		EXPECT_FLOAT_EQ(positions[i].y, targets[i].y);
	}

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
}

// =============================================================================
// Contact Event Collector Tests
// =============================================================================

TEST_F(BatchTest, ContactEventCollector_CreateDestroy) {
	auto* collector = JPH_ContactEventCollector_Create(64);
	ASSERT_NE(collector, nullptr);
	EXPECT_EQ(JPH_ContactEventCollector_GetEventCount(collector), 0u);
	JPH_ContactEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ContactEventCollector_CollisionGeneratesEvents) {
	auto* collector = JPH_ContactEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);
	JPH_ContactEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	// Create floor
	JPH_BodyID floorId = CreateStaticFloor();

	// Create sphere that will fall onto the floor
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 2.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);

	// Step multiple times to let sphere fall and collide
	for (int i = 0; i < 120; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	uint32_t eventCount = JPH_ContactEventCollector_GetEventCount(collector);
	EXPECT_GT(eventCount, 0u);

	// Drain events
	JPH_ContactEventData events[256];
	uint32_t drained = JPH_ContactEventCollector_DrainEvents(collector, events, 256);
	EXPECT_EQ(drained, eventCount);

	// Verify event data
	bool foundAdded = false;
	for (uint32_t i = 0; i < drained; ++i) {
		if (events[i].eventType == JPH_ContactEventType_Added) {
			foundAdded = true;
			// One body should be the sphere, the other the floor
			bool hasSphere = (events[i].body1ID == sphereId || events[i].body2ID == sphereId);
			bool hasFloor = (events[i].body1ID == floorId || events[i].body2ID == floorId);
			EXPECT_TRUE(hasSphere);
			EXPECT_TRUE(hasFloor);
			EXPECT_GT(events[i].pointCount, 0u);
		}
	}
	EXPECT_TRUE(foundAdded);

	// After drain, count should be 0
	EXPECT_EQ(JPH_ContactEventCollector_GetEventCount(collector), 0u);

	// Cleanup
	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ContactEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ContactEventCollector_EventFilter) {
	auto* collector = JPH_ContactEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);

	// Only collect Added events (bit 0)
	JPH_ContactEventCollector_SetEventFilter(collector, 1u << JPH_ContactEventType_Added);
	JPH_ContactEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 2.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	for (int i = 0; i < 120; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	JPH_ContactEventData events[256];
	uint32_t drained = JPH_ContactEventCollector_DrainEvents(collector, events, 256);

	// All events should be Added only (no Persisted or Removed)
	for (uint32_t i = 0; i < drained; ++i) {
		EXPECT_EQ(events[i].eventType, JPH_ContactEventType_Added);
	}

	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ContactEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ContactEventCollector_Clear) {
	auto* collector = JPH_ContactEventCollector_Create(64);
	ASSERT_NE(collector, nullptr);
	JPH_ContactEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 2.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	for (int i = 0; i < 120; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	EXPECT_GT(JPH_ContactEventCollector_GetEventCount(collector), 0u);
	JPH_ContactEventCollector_Clear(collector);
	EXPECT_EQ(JPH_ContactEventCollector_GetEventCount(collector), 0u);

	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ContactEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ContactEventCollector_ContactRemoved) {
	auto* collector = JPH_ContactEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);
	JPH_ContactEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 2.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);

	// Let it collide
	for (int i = 0; i < 60; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}
	JPH_ContactEventCollector_Clear(collector);

	// Move sphere far away to trigger contact removed
	JPH_RVec3 farPos = {0.0f, 100.0f, 0.0f};
	JPH_Quat identity = {0, 0, 0, 1};
	JPH_BodyInterface_SetPosition(bodyInterface, sphereId, &farPos, JPH_Activation_Activate);
	JPH_Vec3 zeroVel = {0, 0, 0};
	JPH_BodyInterface_SetLinearVelocity(bodyInterface, sphereId, &zeroVel);

	// Step to process removal
	for (int i = 0; i < 5; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	JPH_ContactEventData events[256];
	uint32_t drained = JPH_ContactEventCollector_DrainEvents(collector, events, 256);

	bool foundRemoved = false;
	for (uint32_t i = 0; i < drained; ++i) {
		if (events[i].eventType == JPH_ContactEventType_Removed) {
			foundRemoved = true;
		}
	}
	EXPECT_TRUE(foundRemoved);

	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ContactEventCollector_Destroy(collector);
}

// =============================================================================
// Activation Event Collector Tests
// =============================================================================

TEST_F(BatchTest, ActivationEventCollector_CreateDestroy) {
	auto* collector = JPH_ActivationEventCollector_Create(64);
	ASSERT_NE(collector, nullptr);
	EXPECT_EQ(JPH_ActivationEventCollector_GetEventCount(collector), 0u);
	JPH_ActivationEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ActivationEventCollector_ActivationEvents) {
	auto* collector = JPH_ActivationEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);
	JPH_ActivationEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();

	// Create dynamic bodies that will eventually sleep
	const uint32_t count = 3;
	JPH_BodyID bodyIDs[3];
	for (uint32_t i = 0; i < count; ++i) {
		bodyIDs[i] = CreateDynamicSphere((float)i * 3.0f, 0.6f, 0.0f, 0.5f);
		JPH_BodyInterface_AddBody(bodyInterface, bodyIDs[i], JPH_Activation_Activate);
	}

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);

	// Step many times — bodies should activate initially
	for (int i = 0; i < 10; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	uint32_t eventCount = JPH_ActivationEventCollector_GetEventCount(collector);
	// We should have at least activation events from the initial activation
	// (bodies were added with Activate)
	EXPECT_GT(eventCount, 0u);

	JPH_ActivationEventData events[256];
	uint32_t drained = JPH_ActivationEventCollector_DrainEvents(collector, events, 256);
	EXPECT_EQ(drained, eventCount);

	// Verify event data has valid body IDs
	for (uint32_t i = 0; i < drained; ++i) {
		EXPECT_NE(events[i].bodyID, 0u);
		EXPECT_TRUE(events[i].activated == 0 || events[i].activated == 1);
	}

	// After drain, count should be 0
	EXPECT_EQ(JPH_ActivationEventCollector_GetEventCount(collector), 0u);

	for (uint32_t i = 0; i < count; ++i) {
		JPH_BodyInterface_RemoveBody(bodyInterface, bodyIDs[i]);
		JPH_BodyInterface_DestroyBody(bodyInterface, bodyIDs[i]);
	}
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ActivationEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ActivationEventCollector_Clear) {
	auto* collector = JPH_ActivationEventCollector_Create(64);
	ASSERT_NE(collector, nullptr);
	JPH_ActivationEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID bodyId = CreateDynamicSphere(0.0f, 5.0f, 0.0f);
	JPH_BodyInterface_AddBody(bodyInterface, bodyId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);

	// Should have activation event
	JPH_ActivationEventCollector_Clear(collector);
	EXPECT_EQ(JPH_ActivationEventCollector_GetEventCount(collector), 0u);

	JPH_BodyInterface_RemoveBody(bodyInterface, bodyId);
	JPH_BodyInterface_DestroyBody(bodyInterface, bodyId);
	JPH_ActivationEventCollector_Destroy(collector);
}

TEST_F(BatchTest, ActivationEventCollector_DeactivationAfterSleep) {
	auto* collector = JPH_ActivationEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);
	JPH_ActivationEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();

	// Create a sphere resting on the floor — should eventually deactivate
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 1.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);

	// Step enough times for the body to come to rest and deactivate
	for (int i = 0; i < 600; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	JPH_ActivationEventData events[256];
	uint32_t drained = JPH_ActivationEventCollector_DrainEvents(collector, events, 256);

	// Look for deactivation event
	bool foundDeactivation = false;
	for (uint32_t i = 0; i < drained; ++i) {
		if (events[i].bodyID == sphereId && events[i].activated == 0) {
			foundDeactivation = true;
		}
	}
	EXPECT_TRUE(foundDeactivation);

	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ActivationEventCollector_Destroy(collector);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(BatchTest, BatchOps_ZeroCounts) {
	// All batch operations should handle zero count gracefully
	JPH_BodyInterface_GetPositionsAndRotations(bodyInterface, nullptr, 0, nullptr, nullptr);
	JPH_BodyInterface_SetPositionsAndRotations(bodyInterface, nullptr, 0, nullptr, nullptr, JPH_Activation_Activate);
	JPH_BodyInterface_GetLinearAndAngularVelocities(bodyInterface, nullptr, 0, nullptr, nullptr);
	JPH_BodyInterface_SetLinearAndAngularVelocities(bodyInterface, nullptr, 0, nullptr, nullptr);
	JPH_BodyInterface_GetTransformsAndVelocities(bodyInterface, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
	JPH_BodyInterface_SetPositionRotationAndVelocities(bodyInterface, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
	JPH_BodyInterface_AddForces(bodyInterface, nullptr, 0, nullptr);
	JPH_BodyInterface_AddTorques(bodyInterface, nullptr, 0, nullptr);
	JPH_BodyInterface_AddForcesAndTorques(bodyInterface, nullptr, 0, nullptr, nullptr);
	JPH_BodyInterface_AddImpulses(bodyInterface, nullptr, 0, nullptr);
	JPH_BodyInterface_AddAngularImpulses(bodyInterface, nullptr, 0, nullptr);
	JPH_BodyInterface_MoveKinematics(bodyInterface, nullptr, 0, nullptr, nullptr, 1.0f / 60.0f);
	JPH_BodyInterface_DestroyBodies(bodyInterface, nullptr, 0);
}

TEST_F(BatchTest, ContactEventCollector_DrainWithMaxLessThanAvailable) {
	auto* collector = JPH_ContactEventCollector_Create(256);
	ASSERT_NE(collector, nullptr);
	JPH_ContactEventCollector_SetOnPhysicsSystem(collector, physicsSystem);

	JPH_BodyID floorId = CreateStaticFloor();
	JPH_BodyID sphereId = CreateDynamicSphere(0.0f, 2.0f, 0.0f, 0.5f);
	JPH_BodyInterface_AddBody(bodyInterface, sphereId, JPH_Activation_Activate);

	JPH_PhysicsSystem_OptimizeBroadPhase(physicsSystem);
	for (int i = 0; i < 120; ++i) {
		JPH_PhysicsSystem_Update(physicsSystem, 1.0f / 60.0f, 1, jobSystem);
	}

	uint32_t total = JPH_ContactEventCollector_GetEventCount(collector);
	if (total > 1) {
		// Drain only 1 event
		JPH_ContactEventData oneEvent;
		uint32_t drained = JPH_ContactEventCollector_DrainEvents(collector, &oneEvent, 1);
		EXPECT_EQ(drained, 1u);

		// Remaining should be total - 1
		uint32_t remaining = JPH_ContactEventCollector_GetEventCount(collector);
		EXPECT_EQ(remaining, total - 1);
	}

	JPH_BodyInterface_RemoveBody(bodyInterface, sphereId);
	JPH_BodyInterface_DestroyBody(bodyInterface, sphereId);
	JPH_BodyInterface_RemoveBody(bodyInterface, floorId);
	JPH_BodyInterface_DestroyBody(bodyInterface, floorId);
	JPH_ContactEventCollector_Destroy(collector);
}
