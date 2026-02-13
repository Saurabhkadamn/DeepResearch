"""
Deep Research - Virtual File System
In-memory filesystem for context management.
Agents write findings to files, report writer reads them.
Prevents context overflow on long research tasks.
"""


class VirtualFileSystem:
    """Simple in-memory filesystem. Keys are paths, values are content."""

    def __init__(self):
        self._files: dict[str, str] = {}

    def write(self, path: str, content: str) -> str:
        """Write content to a virtual file."""
        self._files[path] = content
        return f"Written {len(content)} chars to {path}"

    def read(self, path: str) -> str:
        """Read content from a virtual file."""
        if path not in self._files:
            return f"File not found: {path}"
        return self._files[path]

    def append(self, path: str, content: str) -> str:
        """Append content to a virtual file."""
        existing = self._files.get(path, "")
        self._files[path] = existing + "\n" + content if existing else content
        return f"Appended {len(content)} chars to {path}"

    def ls(self, directory: str = "/") -> list[str]:
        """List files in a directory."""
        return [p for p in self._files.keys() if p.startswith(directory)]

    def exists(self, path: str) -> bool:
        return path in self._files

    def delete(self, path: str) -> bool:
        if path in self._files:
            del self._files[path]
            return True
        return False

    def read_all_findings(self) -> dict[str, str]:
        """Read all files under /findings/ — used by synthesis and report writer."""
        return {
            path: content
            for path, content in self._files.items()
            if path.startswith("/findings/")
        }

    def to_dict(self) -> dict:
        return dict(self._files)

    @classmethod
    def from_dict(cls, data: dict) -> "VirtualFileSystem":
        vfs = cls()
        vfs._files = dict(data)
        return vfs
