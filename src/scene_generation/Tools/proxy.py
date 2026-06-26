# Helper for pipe-based inter-process communication
import queue
import time
import threading
import sys
from multiprocessing.connection import Client


def _pipe_addresses_for_id(pipe_id: str = ""):
    """Return (in_addr, out_addr, family) with an optional *pipe_id* suffix."""
    suffix = f"_{pipe_id}" if pipe_id else ""
    if sys.platform == "win32":
        default_in = rf"\\.\pipe\isaacsim_cmd_pipe{suffix}"
        default_out = rf"\\.\pipe\isaacsim_out_pipe{suffix}"
        family = 'AF_PIPE'
    else:
        default_in = f"/tmp/isaacsim_cmd_pipe_new{suffix}"
        default_out = f"/tmp/isaacsim_out_pipe_new{suffix}"
        family = 'AF_UNIX'
    return default_in, default_out, family


# Keep legacy module-level constants for backward compatibility (pipe_id="").
if sys.platform == "win32":
    DEFAULT_IN_ADDR = r"\\.\pipe\isaacsim_cmd_pipe"
    DEFAULT_OUT_ADDR = r"\\.\pipe\isaacsim_out_pipe"
    FAMILY = 'AF_PIPE'
else:
    DEFAULT_IN_ADDR = "/tmp/isaacsim_cmd_pipe_new"
    DEFAULT_OUT_ADDR = "/tmp/isaacsim_out_pipe_new"
    FAMILY = 'AF_UNIX'


class IOPipe():
    def __init__(self, pipe_id: str = "", in_address: str = None, out_address: str = None):
        """Create an IOPipe client that talks to a specific Isaac Sim instance.

        Args:
            pipe_id: Instance identifier used to derive pipe addresses.
                When non-empty, ``_{pipe_id}`` is appended to the default pipe
                names so that multiple scene-generator / Isaac-Sim pairs can run
                in parallel.  Ignored when *in_address* and *out_address* are
                both provided explicitly.
            in_address: Explicit command pipe address (overrides *pipe_id*).
            out_address: Explicit output pipe address (overrides *pipe_id*).
        """
        # Resolve effective family from explicit addresses or platform.
        if in_address and out_address:
            self.in_addr = in_address
            self.out_addr = out_address
        else:
            derived_in, derived_out, _ = _pipe_addresses_for_id(pipe_id)
            self.in_addr = in_address or derived_in
            self.out_addr = out_address or derived_out

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Internal state flags
        self._running = True
        self._threads_started = False
        self._threads_lock = threading.Lock()
        self._out_ready = threading.Event()
        self._in_ready = threading.Event()
        self._output_thread = None
        self._input_thread = None

    def stop(self):
        """Stop background retry loops and mark channels as not ready."""
        self._running = False
        self._out_ready.clear()
        self._in_ready.clear()
        with self._threads_lock:
            self._threads_started = False

    def output_receiver_thread(self, q: queue.Queue, address: str, ready_evt: threading.Event | None = None):
        """Client: connect to Isaac Sim's output end and receive data."""
        conn = None
        reconnect_backoff = 0.5  # Slightly longer client reconnect interval to reduce CPU usage

        while self._running:
            if conn is None:
                try:
                    # Attempt to establish connection
                    conn = Client(address, family=FAMILY)
                    print(f"[Client] Connected to output pipe: {address}")
                    if ready_evt:
                        ready_evt.set()
                except Exception:
                    # The server may not be running yet; wait silently
                    if ready_evt:
                        ready_evt.clear()
                    time.sleep(reconnect_backoff)
                    continue

            try:
                # recv is blocking
                msg = conn.recv()
                q.put(msg)
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[Client] Output connection lost ({type(e).__name__}). Retrying...")
                if ready_evt:
                    ready_evt.clear()
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                time.sleep(reconnect_backoff)
            except Exception as e:
                print(f"[Client] Unexpected output receiver error: {e}")
                if ready_evt:
                    ready_evt.clear()
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
                conn = None
                time.sleep(reconnect_backoff)

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def input_sender_thread(self, q: queue.Queue, address: str, ready_evt: threading.Event | None = None):
        """Client: connect to Isaac Sim's input end and send commands."""
        conn = None
        reconnect_backoff = 0.5

        while self._running:
            if conn is None:
                try:
                    conn = Client(address, family=FAMILY)
                    print(f"[Client] Connected to command pipe: {address}")
                    if ready_evt:
                        ready_evt.set()
                except Exception:
                    if ready_evt:
                        ready_evt.clear()
                    time.sleep(reconnect_backoff)
                    continue

            try:
                # Get command from queue with timeout to allow checking conn status
                try:
                    msg = q.get(timeout=1.0)
                except queue.Empty:
                    continue

                try:
                    conn.send(msg)
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                    print(f"[Client] Command connection lost. Re-queueing message.")
                    # Send failed; put the message back at the head of the queue
                    try:
                        q.put_nowait(msg)
                    except Exception:
                        pass

                    if ready_evt:
                        ready_evt.clear()
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                    time.sleep(reconnect_backoff)
            except Exception as e:
                print(f"[Client] Unexpected error in sender: {e}")
                time.sleep(reconnect_backoff)

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def setup(self, timeout=20.0):
        """Start background threads and wait for connections to be ready."""
        self._running = True
        with self._threads_lock:
            output_alive = self._output_thread is not None and self._output_thread.is_alive()
            input_alive = self._input_thread is not None and self._input_thread.is_alive()

            if not output_alive:
                self._output_thread = threading.Thread(
                    target=self.output_receiver_thread,
                    args=(self.output_queue, self.out_addr, self._out_ready),
                    daemon=True,
                )
                self._output_thread.start()

            if not input_alive:
                self._input_thread = threading.Thread(
                    target=self.input_sender_thread,
                    args=(self.input_queue, self.in_addr, self._in_ready),
                    daemon=True,
                )
                self._input_thread.start()

            if self._output_thread is not None and self._input_thread is not None:
                self._threads_started = True

        print(f"Waiting for Isaac Sim pipes ({FAMILY})...")

        # Wait for connections to succeed
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._out_ready.is_set() and self._in_ready.is_set():
                print("IO Pipe Ready (Connected to Isaac Sim)")
                return True
            time.sleep(0.1)

        print("Warning: Setup timeout. Threads will continue to retry in background.")
        return False

    def send_to_is(self, command: str):
        self.input_queue.put(command)

    def recv_from_is(self, timeout=10) -> str:
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

if __name__ == "__main__":
    iopipe = IOPipe()
    iopipe.setup()
    for i in range(5):
        iopipe.send_to_is("Hello {i}".format(i = i))
        print(iopipe.recv_from_is())