import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestIgnoreContextManager(JitTestCase):
    def test_with_ignore_context_manager_with_inp_out(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                with objmode(a="inp:int", b="inp:int", c="out:int", d="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                    d = 9
                return c + d
        s = torch.jit.script(A())
        self.assertEqual(s(), 20)

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                with objmode(a="inp:int", b="inp:int", c="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                return c
        s = torch.jit.script(B())
        self.assertEqual(s(), 11)

        class C(torch.nn.Module):
            def __init__(self):
                super(C, self).__init__()
            def forward(self):
                a: int = 4
                b: int = 5
                with objmode(a="inp:int", b="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    b = l[0] + a
                return b
        s = torch.jit.script(C())
        self.assertEqual(s(), 6)

    def test_with_ignore_context_manager_with_just_inp(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()
            def forward(self):
                a: int = 4
                b: int = 5
                with objmode(a="inp:int", b="inp:int"):
                    l = [2 + b for i in range(a) if i > 2]
                return a
        s = torch.jit.script(A())
        self.assertEqual(s(), 9)

    def test_with_ignore_context_manager_wrong_name(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                with objdfjehfeh(a="inp:int", b="inp:int", c="out:int", d="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                    d = 9
                return c + d
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError,
                                    "Context manager with name objdfjehfeh is not supported"):

            s = torch.jit.script(A())
