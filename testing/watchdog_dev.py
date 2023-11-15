import subprocess
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

PYTHON_ROOTS = ["dino", "testing", "tests"]


class TestRunner(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".py"):
            return
        if event.event_type in ("created", "modified"):
            subprocess.run(["black", *PYTHON_ROOTS])
            subprocess.run(["isort", *PYTHON_ROOTS])
            subprocess.run(["python", "-m", "unittest", "discover", "."])


if __name__ == "__main__":
    event_handler = TestRunner()
    observer = Observer()
    for root in PYTHON_ROOTS:
        observer.schedule(event_handler, root, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
