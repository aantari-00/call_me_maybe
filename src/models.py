from typing import Literal
from pydantic import BaseModel


class ParamSchema(BaseModel):
    type: Literal["number", "string", "boolean", "integer"]


class ReturnSchema(BaseModel):
    type: str


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamSchema]
    returns: ReturnSchema


class PromptItem(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, object]
