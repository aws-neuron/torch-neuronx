#include "Stream.h"

#include <structmember.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_neuronx/csrc/c10/neuron/NeuronEvent.h"
#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"

namespace {
constexpr int DEFAULT_STREAM_PRIORITY = 0;
constexpr c10::DeviceType NEURON_DEVICE_TYPE = c10::DeviceType::PrivateUse1;
}  // anonymous namespace

PyObject* THNPStreamClass = nullptr;
PyObject* THNPEventClass = nullptr;

// Forward declarations for RAII helper functions
static THNPStream* create_stream_object(at::neuron::NeuronStream stream);
static THNPEvent* create_event_object(at::neuron::NeuronEvent event);

THNPStream::THNPStream(at::neuron::NeuronStream stream) : neuron_stream(std::move(stream)) {
  // Initialize base THPStream fields from the moved stream
  stream_id = neuron_stream.id();
  device_index = neuron_stream.device_index();
  device_type = static_cast<int64_t>(neuron_stream.device().type());
}

THNPStream::~THNPStream() = default;

THNPEvent::~THNPEvent() = default;

static PyObject* THNPStream_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({
      "Stream(Device? device=None, *, int64_t priority=0)",
      "Stream(*, int64_t stream_id, int64_t device_index, int64_t device_type)",
  });

  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::neuron::NeuronStream stream;

  if (r.idx == 0) {
    c10::DeviceIndex device_index = at::neuron::current_device();

    if (!r.isNone(0)) {
      auto device = r.device(0);
      TORCH_CHECK(device.type() == NEURON_DEVICE_TYPE, "Expected Neuron device, got: ", device);
      device_index = device.index();
    }

    int priority = r.toInt64(1);
    stream = at::neuron::NeuronStream::createStream(device_index, priority);
  } else {
    c10::StreamId stream_id = r.toInt64(0);
    c10::DeviceIndex device_index = r.toInt64(1);
    int device_type = r.toInt64(2);

    TORCH_CHECK(device_type == static_cast<int>(NEURON_DEVICE_TYPE),
                "Expected Neuron device type, got: ", device_type);
    stream = at::neuron::NeuronStream(stream_id, device_index, at::neuron::NeuronStream::UNCHECKED);
  }

  return (PyObject*)create_stream_object(std::move(stream));
  END_HANDLE_TH_ERRORS
}

static void THNPStream_dealloc(THNPStream* self) {
  self->neuron_stream.~NeuronStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPStream_query(THNPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  bool is_complete = self->neuron_stream.query();
  return PyBool_FromLong(is_complete);
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_synchronize(THNPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    self->neuron_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_wait_event(THNPStream* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"wait_event(PyObject* event)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  PyObject* event_obj = r.pyobject(0);
  TORCH_CHECK(THNPEvent_Check(event_obj), "Expected a Neuron event");

  auto event = THNPEvent_Unpack(event_obj);

  {
    pybind11::gil_scoped_release no_gil;
    self->neuron_stream.wait_event(event);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_wait_stream(THNPStream* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"wait_stream(PyObject* stream)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  PyObject* stream_obj = r.pyobject(0);
  TORCH_CHECK(THNPStream_Check(stream_obj), "Expected a Neuron stream");

  auto other_stream = THNPStream_Unpack(stream_obj);

  {
    pybind11::gil_scoped_release no_gil;
    self->neuron_stream.wait_stream(other_stream);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_record_event(THNPStream* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"record_event(PyObject* event=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  at::neuron::NeuronEvent event;
  if (r.isNone(0)) {
    // Create new event
    event = at::neuron::NeuronEvent();
  } else {
    PyObject* event_obj = r.pyobject(0);
    TORCH_CHECK(THNPEvent_Check(event_obj), "Expected a Neuron event");
    // For existing events, we need to record on them directly
    auto existing_event = THNPEvent_Unpack(event_obj);
    existing_event.record(self->neuron_stream);
    Py_INCREF(event_obj);
    return event_obj;
  }

  {
    pybind11::gil_scoped_release no_gil;
    event.record(self->neuron_stream);
  }

  return THNPEvent_Wrap(std::move(event));
  END_HANDLE_TH_ERRORS
}

// Stream property getters
static PyObject* THNPStream_get_priority(THNPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(self->neuron_stream.priority());
  END_HANDLE_TH_ERRORS
}

// Stream properties table
static PyGetSetDef THNPStream_properties[] = {
    {"priority", (getter)THNPStream_get_priority, nullptr, nullptr, nullptr}, {nullptr}};

static PyMethodDef THNPStream_methods[] = {
    {(char*)"query", (PyCFunction)THNPStream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THNPStream_synchronize, METH_NOARGS, nullptr},
    {(char*)"wait_event", (PyCFunction)THNPStream_wait_event, METH_VARARGS, nullptr},
    {(char*)"wait_stream", (PyCFunction)THNPStream_wait_stream, METH_VARARGS, nullptr},
    {(char*)"record_event", (PyCFunction)THNPStream_record_event, METH_VARARGS, nullptr},
    {nullptr}  // Sentinel
};

static PyTypeObject THNPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._NeuronStreamBase", /* tp_name */
    sizeof(THNPStream),                                             /* tp_basicsize */
    0,                                                              /* tp_itemsize */
    (destructor)THNPStream_dealloc,                                 /* tp_dealloc */
    0,                                                              /* tp_vectorcall_offset */
    0,                                                              /* tp_getattr */
    0,                                                              /* tp_setattr */
    0,                                                              /* tp_reserved */
    0,                                                              /* tp_repr */
    0,                                                              /* tp_as_number */
    0,                                                              /* tp_as_sequence */
    0,                                                              /* tp_as_mapping */
    0,                                                              /* tp_hash  */
    0,                                                              /* tp_call */
    0,                                                              /* tp_str */
    0,                                                              /* tp_getattro */
    0,                                                              /* tp_setattro */
    0,                                                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                       /* tp_flags */
    nullptr,                                                        /* tp_doc */
    0,                                                              /* tp_traverse */
    0,                                                              /* tp_clear */
    0,                                                              /* tp_richcompare */
    0,                                                              /* tp_weaklistoffset */
    0,                                                              /* tp_iter */
    0,                                                              /* tp_iternext */
    THNPStream_methods,                                             /* tp_methods */
    0,                                                              /* tp_members */
    THNPStream_properties,                                          /* tp_getset */
    0,                                                              /* tp_base */
    0,                                                              /* tp_dict */
    0,                                                              /* tp_descr_get */
    0,                                                              /* tp_descr_set */
    0,                                                              /* tp_dictoffset */
    0,                                                              /* tp_init */
    0,                                                              /* tp_alloc */
    THNPStream_pynew,                                               /* tp_new */
};

// Event Python methods
static PyObject* THNPEvent_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({
      "Event(*, bool enable_timing=False, bool blocking=False)",
  });

  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  bool enable_timing = r.toBool(0);
  bool blocking = r.toBool(1);

  // Create new event
  auto event = at::neuron::NeuronEvent(enable_timing, blocking);

  // Create Python object using RAII helper
  return (PyObject*)create_event_object(std::move(event));
  END_HANDLE_TH_ERRORS
}

static void THNPEvent_dealloc(THNPEvent* self) {
  // Destroy the NeuronEvent
  self->neuron_event.~NeuronEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPEvent_query(THNPEvent* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->neuron_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_synchronize(THNPEvent* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    self->neuron_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_record(THNPEvent* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"record(PyObject* stream=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  at::neuron::NeuronStream stream;
  if (r.isNone(0)) {
    stream = at::neuron::getCurrentNeuronStream();
  } else {
    PyObject* stream_obj = r.pyobject(0);
    TORCH_CHECK(THNPStream_Check(stream_obj), "Expected a Neuron stream");
    stream = THNPStream_Unpack(stream_obj);
  }

  {
    pybind11::gil_scoped_release no_gil;
    self->neuron_event.record(stream);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_wait(THNPEvent* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"wait(PyObject* stream=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  at::neuron::NeuronStream stream;
  if (r.isNone(0)) {
    stream = at::neuron::getCurrentNeuronStream();
  } else {
    PyObject* stream_obj = r.pyobject(0);
    TORCH_CHECK(THNPStream_Check(stream_obj), "Expected a Neuron stream");
    stream = THNPStream_Unpack(stream_obj);
  }

  {
    pybind11::gil_scoped_release no_gil;
    stream.wait_event(self->neuron_event);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_elapsed_time(THNPEvent* self, PyObject* args) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({"elapsed_time(PyObject* end_event)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, nullptr, parsed_args);

  PyObject* end_event_obj = r.pyobject(0);
  TORCH_CHECK(THNPEvent_Check(end_event_obj), "Expected a Neuron event");

  auto end_event = THNPEvent_Unpack(end_event_obj);

  float elapsed_ms = self->neuron_event.elapsed_time(end_event);
  return PyFloat_FromDouble(elapsed_ms);
  END_HANDLE_TH_ERRORS
}

// Event property getters
static PyObject* THNPEvent_get_device(THNPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  c10::DeviceIndex device_index = self->neuron_event.device_index();
  if (device_index < 0) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(c10::Device(NEURON_DEVICE_TYPE, device_index));
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_get_neuron_event(THNPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  // Return a pointer representation of the event
  return PyLong_FromVoidPtr(self->neuron_event.get_impl().get());
  END_HANDLE_TH_ERRORS
}

// Event properties table
static PyGetSetDef THNPEvent_properties[] = {
    {"device", (getter)THNPEvent_get_device, nullptr, nullptr, nullptr},
    {"neuron_event", (getter)THNPEvent_get_neuron_event, nullptr, nullptr, nullptr},
    {nullptr}};

// Event methods table
static PyMethodDef THNPEvent_methods[] = {
    {(char*)"query", (PyCFunction)THNPEvent_query, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THNPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"record", (PyCFunction)THNPEvent_record, METH_VARARGS, nullptr},
    {(char*)"wait", (PyCFunction)THNPEvent_wait, METH_VARARGS, nullptr},
    {(char*)"elapsed_time", (PyCFunction)THNPEvent_elapsed_time, METH_VARARGS, nullptr},
    {nullptr}};

// Event type definition
static PyTypeObject THNPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._NeuronEventBase", /* tp_name */
    sizeof(THNPEvent),                                             /* tp_basicsize */
    0,                                                             /* tp_itemsize */
    (destructor)THNPEvent_dealloc,                                 /* tp_dealloc */
    0,                                                             /* tp_vectorcall_offset */
    0,                                                             /* tp_getattr */
    0,                                                             /* tp_setattr */
    0,                                                             /* tp_reserved */
    0,                                                             /* tp_repr */
    0,                                                             /* tp_as_number */
    0,                                                             /* tp_as_sequence */
    0,                                                             /* tp_as_mapping */
    0,                                                             /* tp_hash  */
    0,                                                             /* tp_call */
    0,                                                             /* tp_str */
    0,                                                             /* tp_getattro */
    0,                                                             /* tp_setattro */
    0,                                                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                      /* tp_flags */
    nullptr,                                                       /* tp_doc */
    0,                                                             /* tp_traverse */
    0,                                                             /* tp_clear */
    0,                                                             /* tp_richcompare */
    0,                                                             /* tp_weaklistoffset */
    0,                                                             /* tp_iter */
    0,                                                             /* tp_iternext */
    THNPEvent_methods,                                             /* tp_methods */
    0,                                                             /* tp_members */
    THNPEvent_properties,                                          /* tp_getset */
    0,                                                             /* tp_base */
    0,                                                             /* tp_dict */
    0,                                                             /* tp_descr_get */
    0,                                                             /* tp_descr_set */
    0,                                                             /* tp_dictoffset */
    0,                                                             /* tp_init */
    0,                                                             /* tp_alloc */
    THNPEvent_pynew,                                               /* tp_new */
};

static THNPStream* create_stream_object(at::neuron::NeuronStream stream) {
  THNPStream* self = (THNPStream*)THNPStreamType.tp_alloc(&THNPStreamType, 0);
  if (self != nullptr) {
    try {
      new (&self->neuron_stream) at::neuron::NeuronStream(std::move(stream));
      self->stream_id = self->neuron_stream.id();
      self->device_index = self->neuron_stream.device_index();
      self->device_type = static_cast<int64_t>(self->neuron_stream.device().type());
    } catch (...) {
      Py_TYPE(self)->tp_free((PyObject*)self);
      throw;
    }
  }
  return self;
}

static THNPEvent* create_event_object(at::neuron::NeuronEvent event) {
  THNPEvent* self = (THNPEvent*)THNPEventType.tp_alloc(&THNPEventType, 0);
  if (self != nullptr) {
    try {
      new (&self->neuron_event) at::neuron::NeuronEvent(std::move(event));
    } catch (...) {
      Py_TYPE(self)->tp_free((PyObject*)self);
      throw;
    }
  }
  return self;
}

void THNPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THNPStreamType.tp_base = THPStreamClass;
  THNPStreamClass = (PyObject*)&THNPStreamType;
  if (PyType_Ready(&THNPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THNPStreamType);
  if (PyModule_AddObject(module, "_NeuronStreamBase", (PyObject*)&THNPStreamType) < 0) {
    throw python_error();
  }
}

void THNPEvent_init(PyObject* module) {
  THNPEventClass = (PyObject*)&THNPEventType;
  if (PyType_Ready(&THNPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THNPEventType);
  if (PyModule_AddObject(module, "_NeuronEventBase", (PyObject*)&THNPEventType) < 0) {
    throw python_error();
  }
}

at::neuron::NeuronStream THNPStream_Unpack(PyObject* obj) {
  TORCH_CHECK(THNPStream_Check(obj), "Expected a Neuron stream");
  return ((THNPStream*)obj)->neuron_stream;
}

at::neuron::NeuronEvent THNPEvent_Unpack(PyObject* obj) {
  TORCH_CHECK(THNPEvent_Check(obj), "Expected a Neuron event");
  return ((THNPEvent*)obj)->neuron_event;
}

PyObject* THNPStream_Wrap(at::neuron::NeuronStream stream) {
  return (PyObject*)create_stream_object(std::move(stream));
}

PyObject* THNPEvent_Wrap(at::neuron::NeuronEvent event) {
  return (PyObject*)create_event_object(std::move(event));
}
