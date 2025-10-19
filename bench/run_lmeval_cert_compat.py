"""Compatibility wrapper for running bench.run_lmeval_cert with varying lm-eval versions.
This script attempts to ensure TaskManager-like API exists on lm_eval.tasks so the
existing `bench.run_lmeval_cert` can import and run unchanged. It then delegates to
run_lmeval_cert.main().
"""
import importlib
import sys
import types

def ensure_taskmanager_fallback():
    try:
        import lm_eval.tasks as _letasks
    except Exception:
        return

    # If TaskManager already exists, nothing to do
    if hasattr(_letasks, 'TaskManager'):
        return

    # Build a minimal TaskManager replacement if possible
    class _FallbackTaskManager:
        def __init__(self):
            # try to use an available task_index if present
            self.task_index = {}
            if hasattr(_letasks, 'task_index'):
                try:
                    self.task_index = dict(_letasks.task_index)
                except Exception:
                    # leave empty
                    self.task_index = {}

    # Attach fallback to module
    try:
        setattr(_letasks, 'TaskManager', _FallbackTaskManager)
    except Exception:
        pass

def main():
    ensure_taskmanager_fallback()
    # Re-import the target module fresh
    try:
        mod = importlib.import_module('bench.run_lmeval_cert')
        # If module exposes main(), call it
        if hasattr(mod, 'main'):
            mod.main()
        else:
            # Fallback: run as module
            import runpy
            runpy.run_module('bench.run_lmeval_cert', run_name='__main__')
    except Exception as e:
        print('Compat wrapper failed to run bench.run_lmeval_cert:', e)
        raise

if __name__ == '__main__':
    main()
