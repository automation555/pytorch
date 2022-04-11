from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

# Must happen before importing caffe2.python.*
import caffe2.python.fakelowp.init_shared_libs  # noqa
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.fakelowp.test_utils import print_test_debug_info
import caffe2.python.serialized_test.serialized_test_util as serial

workspace.GlobalInit(
    [
        "caffe2",
        "--glow_global_fp16=1",
        "--glow_global_fused_scale_offset_fp16=1",
        "--glow_global_force_sls_fp16_accum=1",
    ]
)

class Fusions(serial.SerializedTestCase):
    @given(scale=st.floats(1e-4, 20), zp=st.integers(-20, 20))
    @settings(deadline=None, max_examples=1000)
    def test_quantize(self, scale, zp):
        workspace.ResetWorkspace()
        pred_net = caffe2_pb2.NetDef()
        pred_net.name = "pred"
        pred_net.external_input.append("X")
        pred_net.external_output.append("Y_q")
        x_scale = scale # 0.5

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Tanh", ["X"], ["Y"]
            )
        )

        pred_net.op.add().CopyFrom(
            core.CreateOperator(
                "Int8Quantize", ["Y"], ["Y_q"], Y_scale=x_scale, Y_zero_point=zp
            )
        )

        X = np.linspace(-20, 20, 100000, dtype=np.float32)
        pred_net_onnxified = onnxifi_caffe2_net(
            pred_net,
            {"X": X.shape},
            debug=True,
            adjust_batch=False,
            use_onnx=False,
        )
        num_onnxified_ops = sum(
            1 if o.type == "Onnxifi" else 0 for o in pred_net_onnxified.op
        )
        np.testing.assert_equal(num_onnxified_ops, 1)
        workspace.FeedBlob("X", X)
        workspace.CreateNet(pred_net_onnxified)
        workspace.RunNet(pred_net_onnxified.name)
        Y_glow = workspace.FetchInt8Blob("Y_q")

        pred_net.op[0].type = "TanhFakeFp16NNPI"
        pred_net.op[1].type = "Int8QuantizeNNPI"
        workspace.RunNetOnce(pred_net)
        Y_ref = workspace.FetchInt8Blob("Y_q")

        print(Y_glow)
        print(Y_ref)

        diff = {}
        for i in range(Y_ref.data.shape[0]):
            if Y_ref.data[i] != Y_glow.data[i]:
                diff[i] = [X[i], Y_ref.data[i], Y_glow.data[i]]
        print(diff)
        np.testing.assert_equal(Y_ref.data, Y_glow.data)
