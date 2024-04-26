from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class cont_request(_message.Message):
    __slots__ = ["step", "world_rank"]
    STEP_FIELD_NUMBER: _ClassVar[int]
    WORLD_RANK_FIELD_NUMBER: _ClassVar[int]
    step: int
    world_rank: int
    def __init__(self, step: _Optional[int] = ..., world_rank: _Optional[int] = ...) -> None: ...

class cont_response(_message.Message):
    __slots__ = ["active_list", "status"]
    ACTIVE_LIST_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    active_list: _containers.RepeatedScalarFieldContainer[int]
    status: int
    def __init__(self, active_list: _Optional[_Iterable[int]] = ..., status: _Optional[int] = ...) -> None: ...

class hook_request(_message.Message):
    __slots__ = ["step", "world_rank"]
    STEP_FIELD_NUMBER: _ClassVar[int]
    WORLD_RANK_FIELD_NUMBER: _ClassVar[int]
    step: int
    world_rank: int
    def __init__(self, step: _Optional[int] = ..., world_rank: _Optional[int] = ...) -> None: ...

class hook_response(_message.Message):
    __slots__ = ["active_list"]
    ACTIVE_LIST_FIELD_NUMBER: _ClassVar[int]
    active_list: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, active_list: _Optional[_Iterable[int]] = ...) -> None: ...
