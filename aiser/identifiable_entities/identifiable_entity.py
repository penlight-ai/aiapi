import typing


class IdentifiableEntity:
    def __init__(self, entity_id: str):
        super().__init__()
        self._id = entity_id

    def accepts_id(self, entity_id: str) -> bool:
        return self._id == entity_id

    def get_id(self) -> typing.Optional[str]:
        return self._id

    def get_id_or_raise(self) -> str:
        if not self._id:
            raise ValueError("Entity ID is not set")
        return self._id

