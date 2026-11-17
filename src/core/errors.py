class ProcessingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ReviewRequired(ProcessingError):
    """O documento não pode ser processado com segurança sem revisão."""
