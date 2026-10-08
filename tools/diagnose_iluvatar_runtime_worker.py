#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one isolated CoreX diagnostic; the caller must enforce a timeout.

Only ``--probe torch`` imports Torch. The driver and runtime probes use the
CUDA-compatible C ABI exposed by CoreX, without importing Triton or FlagGems.
``--device`` is an ordinal in the current process's visible device list.
"""

import argparse
import ctypes
import faulthandler
import json
import os
import sys
import time
import traceback


class ProbeError(RuntimeError):
    def __init__(self, category, message, **details):
        super().__init__(message)
        self.category = category
        self.details = details


class Reporter:
    def __init__(self, probe, device):
        self.probe = probe
        self.device = device
        self.start = time.monotonic()

    def emit(self, stage, event, **details):
        print(
            json.dumps(
                {
                    "probe": self.probe,
                    "device": self.device,
                    "pid": os.getpid(),
                    "elapsed_s": round(time.monotonic() - self.start, 6),
                    "stage": stage,
                    "event": event,
                    **details,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def maps(self, stage="libraries"):
        try:
            paths = set()
            with open("/proc/self/maps", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip().split(maxsplit=5)
                    if len(fields) == 6 and ".so" in fields[5]:
                        paths.add(fields[5])
            self.emit(stage, "INFO", paths=sorted(paths))
        except OSError as exc:
            self.emit(stage, "INFO", maps_error=str(exc))

    def call(self, stage, fn, *args):
        self.emit(stage, "BEGIN")
        result = fn(*args)
        self.emit(stage, "RETURN")
        return result


def load_library(reporter, names):
    errors = []
    for name in names:
        reporter.emit("library.load", "BEGIN", name=name)
        try:
            library = ctypes.CDLL(name)
        except OSError as exc:
            errors.append({"name": name, "error": str(exc)})
            reporter.emit("library.load", "INFO", name=name, error=str(exc))
            continue
        reporter.emit("library.load", "RETURN", name=name)
        reporter.maps()
        return library
    raise ProbeError("LIBRARY_ERROR", "Could not load shared library", attempts=errors)


def bind(library, names, argtypes, restype=ctypes.c_int):
    for name in names:
        try:
            function = getattr(library, name)
        except AttributeError:
            continue
        function.argtypes = argtypes
        function.restype = restype
        return function, name
    raise ProbeError("LIBRARY_ERROR", "Missing C API symbol", symbols=list(names))


def call_api(reporter, name, function, *args, error_text=None):
    reporter.emit(name, "BEGIN")
    code = function(*args)
    reporter.emit(name, "RETURN", return_code=code)
    if code != 0:
        # Persist the numeric error before asking the runtime to describe it.
        reporter.emit(name, "ERROR", category="API_ERROR", return_code=code)
        message = error_text(code) if error_text is not None else None
        raise ProbeError(
            "API_ERROR", name + " failed", api=name, return_code=code, error=message
        )


def device_identity(reporter, library, device, driver=False):
    """Query optional string metadata without relying on a device-properties ABI."""
    pci_symbol = "cuDeviceGetPCIBusId" if driver else "cudaDeviceGetPCIBusId"
    queries = [("pci_bus_id", pci_symbol)]
    if driver:
        queries.append(("name", "cuDeviceGetName"))
    details = {}
    for key, symbol in queries:
        try:
            function, _ = bind(
                library,
                (symbol,),
                [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
            )
        except ProbeError:
            details[key + "_unavailable"] = "symbol unavailable"
            continue
        output = ctypes.create_string_buffer(256)
        reporter.emit(symbol, "BEGIN")
        code = function(output, len(output), device)
        reporter.emit(symbol, "RETURN", return_code=code)
        if code != 0:
            details[key + "_error_code"] = code
        else:
            details[key] = output.value.decode("utf-8", errors="replace")
    reporter.emit("device.identity", "INFO", **details)


def runtime_probe(reporter, device):
    runtime = load_library(
        reporter, ("libcudart.so", "libcudart.so.10.2", "libcudart.so.10.2.89")
    )
    set_device, _ = bind(runtime, ("cudaSetDevice",), [ctypes.c_int])
    free, _ = bind(runtime, ("cudaFree",), [ctypes.c_void_p])
    sync, _ = bind(runtime, ("cudaDeviceSynchronize",), [])
    malloc, _ = bind(
        runtime, ("cudaMalloc",), [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    )
    copy, _ = bind(
        runtime,
        ("cudaMemcpy",),
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    )
    get_error, _ = bind(
        runtime, ("cudaGetErrorString",), [ctypes.c_int], ctypes.c_char_p
    )

    def error_text(code):
        text = get_error(code)
        return text.decode("utf-8", errors="replace") if text else None

    def call(name, function, *args):
        call_api(reporter, name, function, *args, error_text=error_text)

    call("cudaSetDevice", set_device, device)
    call("cudaFree(NULL)", free, None)
    reporter.maps("libraries.after_init")
    call("cudaDeviceSynchronize.initial", sync)
    reporter.emit("runtime.initial_sync", "PASS")
    device_identity(reporter, runtime, device)

    # cudaMemcpyKind values: HostToDevice=1, DeviceToHost=2.
    allocation = ctypes.c_void_p()
    source = ctypes.c_uint32(0x1234ABCD)
    result = ctypes.c_uint32(0)
    size = ctypes.sizeof(source)
    call("cudaMalloc", malloc, ctypes.byref(allocation), size)
    try:
        call("cudaMemcpy.H2D", copy, allocation, ctypes.byref(source), size, 1)
        call("cudaDeviceSynchronize.after_copy", sync)
        call("cudaMemcpy.D2H", copy, ctypes.byref(result), allocation, size, 2)
        call("cudaDeviceSynchronize.after_readback", sync)
        if result.value != source.value:
            raise ProbeError(
                "CHECK_ERROR",
                "Runtime copy round trip changed bytes",
                expected=source.value,
                actual=result.value,
            )
        reporter.emit("runtime.copy_roundtrip", "PASS", value=result.value)
    finally:
        call("cudaFree", free, allocation)


def driver_probe(reporter, device):
    driver = load_library(reporter, ("libcuda.so.1",))
    init, _ = bind(driver, ("cuInit",), [ctypes.c_uint])
    get_device, _ = bind(
        driver, ("cuDeviceGet",), [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    )
    # Explicitly select the three-argument v2 ABI. Newer cuCtxCreate_v3/v4
    # variants have different argument lists and must not be used here.
    create, create_name = bind(
        driver,
        ("cuCtxCreate_v2", "cuCtxCreate"),
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
    )
    sync, _ = bind(driver, ("cuCtxSynchronize",), [])
    destroy, destroy_name = bind(
        driver, ("cuCtxDestroy_v2", "cuCtxDestroy"), [ctypes.c_void_p]
    )
    try:
        get_error, _ = bind(
            driver,
            ("cuGetErrorString",),
            [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        )
    except ProbeError:
        get_error = None

    def error_text(code):
        if get_error is None:
            return None
        message = ctypes.c_char_p()
        error_code = get_error(code, ctypes.byref(message))
        if error_code != 0 or not message.value:
            return None
        return message.value.decode("utf-8", errors="replace")

    def call(name, function, *args):
        call_api(reporter, name, function, *args, error_text=error_text)

    call("cuInit", init, 0)
    selected = ctypes.c_int()
    call("cuDeviceGet", get_device, ctypes.byref(selected), device)
    context = ctypes.c_void_p()
    reporter.emit("driver.context", "INFO", context_kind="private")
    call(create_name, create, ctypes.byref(context), 0, selected.value)
    reporter.maps("libraries.after_init")
    try:
        call("cuCtxSynchronize", sync)
        reporter.emit("driver.initial_sync", "PASS")
        device_identity(reporter, driver, selected.value, driver=True)
    finally:
        call(destroy_name, destroy, context)


def torch_probe(reporter, device):
    reporter.emit("torch.import", "BEGIN")
    try:
        import torch
    except ImportError as exc:
        raise ProbeError("IMPORT_ERROR", str(exc)) from exc
    reporter.emit(
        "torch.import",
        "RETURN",
        version=torch.__version__,
        path=torch.__file__,
        cuda_build=torch.version.cuda,
        python=sys.executable,
    )
    reporter.call("torch.cuda.init", torch.cuda.init)
    reporter.call("torch.cuda.set_device", torch.cuda.set_device, device)
    reporter.maps("libraries.after_init")
    reporter.call("torch.cuda.synchronize.initial", torch.cuda.synchronize, device)
    reporter.emit("torch.initial_sync", "PASS")

    value = reporter.call(
        "torch.allocate",
        lambda: torch.ones(16, dtype=torch.float32, device=f"cuda:{device}"),
    )
    reporter.call("torch.synchronize.after_allocate", torch.cuda.synchronize, device)
    output = reporter.call("torch.add", lambda: value + 2.0)
    reporter.call("torch.synchronize.after_add", torch.cuda.synchronize, device)
    host = reporter.call("torch.copy_to_cpu", output.cpu)
    reporter.call("torch.synchronize.after_readback", torch.cuda.synchronize, device)
    expected = [3.0] * 16
    actual = host.tolist()
    if actual != expected:
        raise ProbeError(
            "CHECK_ERROR", "Torch FP32 add failed", expected=expected, actual=actual
        )
    reporter.emit("torch.fp32_add_readback", "PASS", values=actual)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", choices=("runtime", "driver", "torch"), required=True
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trace-timeout", type=float, default=10)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.trace_timeout < 0:
        parser.error("--trace-timeout must be nonnegative (0 disables traceback)")

    reporter = Reporter(args.probe, args.device)
    if args.trace_timeout:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.trace_timeout, repeat=False)
    reporter.emit(
        "probe",
        "BEGIN",
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH"),
        ld_preload=os.environ.get("LD_PRELOAD"),
    )
    try:
        {"runtime": runtime_probe, "driver": driver_probe, "torch": torch_probe}[
            args.probe
        ](reporter, args.device)
    except ProbeError as exc:
        reporter.emit(
            "probe", "ERROR", category=exc.category, message=str(exc), **exc.details
        )
        reporter.maps("libraries.on_error")
        return 2
    except Exception as exc:
        reporter.emit("probe", "ERROR", category="UNEXPECTED_ERROR", message=repr(exc))
        reporter.maps("libraries.on_error")
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        if args.trace_timeout:
            faulthandler.cancel_dump_traceback_later()
    reporter.maps("libraries.final")
    reporter.emit("probe", "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
