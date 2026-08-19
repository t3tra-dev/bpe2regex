from dataclasses import dataclass


@dataclass(slots=True)
class TokenMatch:
    string: bytes
    token_ids: list[int]
    _token_spans: list[tuple[int, int]]
    path_count: int = 1

    def group(self, key: int | str = 0) -> bytes | None:
        if key == 0:
            return self.string
        if key == "tok":
            if not self._token_spans:
                return None
            start, end = self._token_spans[-1]
            return self.string[start:end]
        raise IndexError(f"unknown group: {key!r}")

    def span(self, key: int | str = 0) -> tuple[int, int]:
        if key == 0:
            return (0, len(self.string))
        if key == "tok":
            return self._token_spans[-1] if self._token_spans else (-1, -1)
        raise IndexError(f"unknown group: {key!r}")

    def captures(self, name: str = "tok") -> list[bytes]:
        if name != "tok":
            raise IndexError(f"unknown group: {name!r}")
        return [self.string[start:end] for start, end in self._token_spans]

    def spans(self, name: str = "tok") -> list[tuple[int, int]]:
        if name != "tok":
            raise IndexError(f"unknown group: {name!r}")
        return list(self._token_spans)
