"""Pipe communication utilities for Isaac Sim standalone app."""

import os
import queue
import re
import sys
import threading
import time
import uuid
from multiprocessing.connection import Listener


def isolated_pipe_id(
    configured_pipe_id=None,
    *,
    process_id=None,
    run_token=None,
):
    """Return a pipe ID owned by one process invocation."""
    namespace = re.sub(
        r'[^A-Za-z0-9_-]+',
        '_',
        str(configured_pipe_id or 'eval'),
    ).strip('_') or 'eval'
    pid = os.getpid() if process_id is None else int(process_id)
    token = re.sub(
        r'[^A-Za-z0-9_-]+',
        '',
        str(run_token or uuid.uuid4().hex[:12]),
    )
    if not token:
        token = uuid.uuid4().hex[:12]
    return f'{namespace}_{pid}_{token}'


def _default_pipe_addresses(pipe_id: str = ""):
    """Return (cmd_addr, out_addr, family) with an optional *pipe_id* suffix.

    When *pipe_id* is non-empty the suffix ``_{pipe_id}`` is appended to each
    pipe name so that multiple Isaac Sim instances can coexist on the same
    machine without colliding on the same named-pipe / Unix-socket addresses.
    """
    suffix = f"_{pipe_id}" if pipe_id else ""
    if sys.platform == "win32":
        default_cmd = rf"\\.\pipe\isaacsim_cmd_pipe{suffix}"
        default_out = rf"\\.\pipe\isaacsim_out_pipe{suffix}"
        default_family = "AF_PIPE"
    else:
        default_cmd = f"/tmp/isaacsim_cmd_pipe_new{suffix}"
        default_out = f"/tmp/isaacsim_out_pipe_new{suffix}"
        default_family = "AF_UNIX"
    return default_cmd, default_out, default_family


class PipeCommunicationServer:
    """Manage command input and output channels over platform-specific pipes."""

    def __init__(
        self,
        input_queue,
        output_queue,
        cmd_pipe_addr=None,
        out_pipe_addr=None,
        family=None,
        pipe_id: str = "",
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue

        default_cmd, default_out, default_family = _default_pipe_addresses(pipe_id)

        self.out_pipe_addr = out_pipe_addr or default_out
        self.cmd_pipe_addr = cmd_pipe_addr or default_cmd
        self.family = family or default_family
        self._running = True

    def stop(self):
        self._running = False

    def _cleanup_socket(self, address):
        if sys.platform != 'win32' and os.path.exists(address):
            try:
                os.remove(address)
                print(f'Cleaned up stale socket: {address}')
            except Exception as exc:
                print(f'Error cleaning up socket {address}: {exc}')

    def _output_sender_thread(self):
        while self._running:
            listener = None
            self._cleanup_socket(self.out_pipe_addr)
            try:
                listener = Listener(self.out_pipe_addr, family=self.family)
                print(f'Output server started at {self.out_pipe_addr}')

                while self._running:
                    conn = None
                    try:
                        conn = listener.accept()
                        print(f'Output client connected: {self.out_pipe_addr}')

                        while self._running:
                            try:
                                sim_output = self.output_queue.get(timeout=1.0)
                            except queue.Empty:
                                continue

                            try:
                                conn.send(sim_output)
                            except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                                print(f'Output connection lost: {exc}')
                                try:
                                    self.output_queue.put_nowait(sim_output)
                                except Exception:
                                    pass
                                break
                    except (OSError, EOFError, ConnectionResetError, BrokenPipeError) as exc:
                        print(f'Output listener error: {exc}. Rebuilding listener...')
                        break
                    except Exception as exc:
                        print(f'Output listener unexpected error: {exc}')
                        time.sleep(0.5)
                    finally:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
            except Exception as exc:
                print(f'CRITICAL: Failed to create output listener: {exc}')
                time.sleep(1)
            finally:
                if listener:
                    try:
                        listener.close()
                    except Exception:
                        pass

    def _pipe_listener_thread(self):
        while self._running:
            listener = None
            self._cleanup_socket(self.cmd_pipe_addr)
            try:
                listener = Listener(self.cmd_pipe_addr, family=self.family)
                print(f'Command server started at {self.cmd_pipe_addr}')

                while self._running:
                    conn = None
                    try:
                        conn = listener.accept()
                        print(f'Command client connected: {self.cmd_pipe_addr}')

                        while self._running:
                            try:
                                msg = conn.recv()
                            except (EOFError, ConnectionResetError, BrokenPipeError):
                                print('Command client disconnected.')
                                break
                            except Exception as exc:
                                print(f'Command recv error: {exc}')
                                break

                            if isinstance(msg, str):
                                self.input_queue.put(msg)
                                if msg == 'quit':
                                    print('Received quit command.')
                                    self.stop()
                                    break
                            else:
                                print(f'Warning: Received non-string data: {type(msg)}')
                    except (OSError, EOFError, ConnectionResetError, BrokenPipeError) as exc:
                        print(f'Command listener error: {exc}. Rebuilding listener...')
                        break
                    except Exception as exc:
                        print(f'Command listener unexpected error: {exc}')
                        time.sleep(0.5)
                    finally:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
            except Exception as exc:
                print(f'CRITICAL: Failed to create command listener: {exc}')
                time.sleep(1)
            finally:
                if listener:
                    try:
                        listener.close()
                    except Exception:
                        pass

    def start(self):
        threading.Thread(target=self._pipe_listener_thread, daemon=True).start()
        threading.Thread(target=self._output_sender_thread, daemon=True).start()
