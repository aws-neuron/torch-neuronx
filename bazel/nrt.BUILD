# BUILD file for Neuron Runtime (NRT)
# This assumes NRT is installed in /opt/aws/neuron/
package(
    default_visibility = [
        "//visibility:public",
    ],
)
cc_library(
    name = "nrt",
    hdrs = glob([
        "include/**/*.h",
    ]),
    includes = ["include"],
    linkopts = [
        "-L/opt/aws/neuron/lib",
        "-lnrt",
        "-Wl,-rpath,/opt/aws/neuron/lib",
    ],
)

cc_import(
    name = "libnrt",
    shared_library = "lib/libnrt.so",
)
