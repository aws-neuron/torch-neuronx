/*
 * Neuron Profiler Registration API
 *
 * This header provides the public registration interface for the Neuron profiler.
 * It has no external dependencies (no libkineto headers) and can be safely included
 * from any compilation unit.
 */

#pragma once

namespace at::neuron {

/**
 * Register the Neuron profiler with libkineto.
 *
 * Can be called multiple times safely (idempotent).
 * Returns true if registration succeeded or was already registered.
 * Returns false if libkineto profiler API is not ready yet.
 */
bool registerNeuronProfiler();

}  // namespace at::neuron
