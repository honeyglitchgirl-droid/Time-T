import numpy as np
from timet.backend import NumpyCpuBackend, get_default_backend, Backend, BackendCapabilities


def test_default_backend_is_numpy_cpu():
    b = get_default_backend()
    assert b.name in ("cpu-numpy", "cpu-simd-openmp")



def test_capabilities_reports_no_gpu():
    caps = get_default_backend().capabilities()
    assert caps.gpu is False
    assert caps.supports_f32 is True
    assert caps.supports_f64 is True
    assert isinstance(caps.device_name, str) and len(caps.device_name) > 0


def test_capabilities_to_json_serializable():
    import json
    caps = get_default_backend().capabilities()
    s = json.dumps(caps.to_json())
    assert ("cpu-numpy" in s or "cpu-simd-openmp" in s)



def test_matmul_correctness():
    b = NumpyCpuBackend()
    a = np.random.rand(3, 4)
    c = np.random.rand(4, 5)
    result = b.matmul(a, c)
    assert np.allclose(result, a @ c)


def test_elementwise_ops():
    b = NumpyCpuBackend()
    a = np.array([1.0, -2.0, 3.0])
    c = np.array([4.0, 5.0, 6.0])
    assert np.allclose(b.elementwise("add", a, c), a + c)
    assert np.allclose(b.elementwise("relu", a), np.maximum(a, 0))


def test_unknown_elementwise_op_raises():
    b = NumpyCpuBackend()
    try:
        b.elementwise("bogus_op", np.array([1.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


class FakeBackend(Backend):
    """Test double proving the Backend ABC is a genuinely swappable interface."""
    name = "fake"

    def capabilities(self):
        return BackendCapabilities("fake", True, False, False, "none", 4, "fake-device")

    def matmul(self, a, b):
        return a @ b

    def elementwise(self, op, *args):
        return args[0]


def test_backend_abc_is_extensible():
    fb = FakeBackend()
    assert fb.capabilities().name == "fake"
    assert fb.matmul(np.eye(2), np.eye(2)).tolist() == np.eye(2).tolist()
