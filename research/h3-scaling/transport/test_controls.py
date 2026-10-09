import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport_control', ROOT / 'transport_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)

class Controls(unittest.TestCase):

    def configure(self, mode='host', initialized=False, env=None):
        fake = types.SimpleNamespace(distributed=types.SimpleNamespace(is_initialized=lambda: initialized))
        env = {'CUDA_VISIBLE_DEVICES': 'a,b,c,d', **control.SEALED} if env is None else env
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {'torch': fake}), patch.object(os, 'write') as write:
            result = control.configure(mode)
            return (result, dict(os.environ), write.call_args)

    def test_only_transport_delta(self):
        peer, _, _ = self.configure('peer')
        host, _, _ = self.configure('host')
        delta = {k: (peer['environment'][k], v) for k, v in host['environment'].items() if peer['environment'][k] != v}
        self.assertEqual(delta, {'NCCL_P2P_DISABLE': ('0', '1')})

    def test_sealed_values_unchanged_and_receipt_bounded(self):
        result, env, write = self.configure()
        self.assertEqual(result['unchanged_seal'], control.SEALED)
        self.assertEqual(write[0][0], 2)
        self.assertLess(len(write[0][1]), 4096)
        self.assertEqual(env['NCCL_DEBUG_FILE'], '')

    def test_refuses_late_or_wrong_width_or_bad_mode(self):
        with self.assertRaises(RuntimeError):
            self.configure(initialized=True)
        with self.assertRaises(RuntimeError):
            self.configure(env={'CUDA_VISIBLE_DEVICES': 'a,b', **control.SEALED})
        with self.assertRaises(ValueError):
            self.configure(mode='unknown')

    def test_refuses_seal_mismatch(self):
        with self.assertRaises(RuntimeError):
            self.configure(env={'CUDA_VISIBLE_DEVICES': 'a,b,c,d', 'NCCL_NVLS_ENABLE': '1', 'NCCL_P2P_LEVEL': 'NVL'})
if __name__ == '__main__':
    unittest.main()
