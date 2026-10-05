from pydantic import BaseModel, ConfigDict

class ContractModel(BaseModel):
    """Shared settings for contract records."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
    )
