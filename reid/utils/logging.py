from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        try:
            # write to original console (stdout)
            if self.console is not None:
                self.console.write(msg)
        except Exception:
            # ignore console write errors
            pass
        if self.file is not None:
            try:
                self.file.write(msg)
            except Exception:
                # ignore file write errors to avoid crashing training
                pass

    def flush(self):
        try:
            if self.console is not None:
                self.console.flush()
        except Exception:
            pass
        if self.file is not None:
            try:
                self.file.flush()
                os.fsync(self.file.fileno())
            except Exception:
                pass

    def close(self):
        # Do NOT close the original console (sys.stdout). Closing sys.stdout
        # can break the interactive session or further prints. Only close
        # the file we opened for logging.
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None
