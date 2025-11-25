#pragma once

#include <c10/core/Stream.h>
#include <torch/csrc/Event.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"

// Neuron-specific stream implementation following THCPStream pattern
struct THNPStream : THPStream {
  at::neuron::NeuronStream neuron_stream;

  THNPStream() = delete;
  explicit THNPStream(at::neuron::NeuronStream stream);
  ~THNPStream();
};

// Neuron-specific event implementation
struct THNPEvent : THPEvent {
  at::neuron::NeuronEvent neuron_event;

  THNPEvent() = delete;
  explicit THNPEvent(at::neuron::NeuronEvent event);
  ~THNPEvent();
};

extern PyObject* THNPStreamClass;
extern PyObject* THNPEventClass;

void THNPStream_init(PyObject* module);
void THNPEvent_init(PyObject* module);

inline bool THNPStream_Check(PyObject* obj) {
  return THNPStreamClass && PyObject_IsInstance(obj, THNPStreamClass);
}

inline bool THNPEvent_Check(PyObject* obj) {
  return THNPEventClass && PyObject_IsInstance(obj, THNPEventClass);
}

// Utility functions for Python bindings
at::neuron::NeuronStream THNPStream_Unpack(PyObject* obj);
at::neuron::NeuronEvent THNPEvent_Unpack(PyObject* obj);

PyObject* THNPStream_Wrap(at::neuron::NeuronStream stream);
PyObject* THNPEvent_Wrap(at::neuron::NeuronEvent event);
