from uuid import uuid4


def new_id(prefix: str) -> str:
    """
    Genera un identificativo univoco leggibile.

    Esempio:
    new_id("resp") -> "resp_a3f91b7c2d10"
    """

    return f"{prefix}_{uuid4().hex[:12]}"